import discord
from discord import Member, Embed, File, SelectOption, ui
from discord.ext import commands
import random
import asyncio
import aiohttp
import openai
import os
import requests
import urllib.parse
import python_weather
from datetime import datetime
from bs4 import BeautifulSoup
import aiofiles
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import logging
from better_profanity import Profanity
from PIL import Image, ImageDraw, ImageFont
import io
import discord.opus

# Bot Configuration and Setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv("TOKENS.env")
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_GPT_TOKEN = os.getenv("OPENAI_GPT_TOKEN")

if not BOT_TOKEN or not OPENAI_GPT_TOKEN:
    logger.critical("Missing BOT_TOKEN or OPENAI_GPT_TOKEN in TOKENS.env")
    exit(1)

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True
bot = commands.Bot(
    command_prefix=".", intents=intents, case_insensitive=True, max_messages=1000
)
bot.remove_command("help")

executor = ThreadPoolExecutor(max_workers=4)
profanity = Profanity()
BOT_IDENTITY = "I am QUARGLE, your AI-powered assistant! I assist users in this Discord server by answering questions, generating ideas, and helping with tasks. I keep answers short, concise and simple"
HISTORY_DIR = "Conversation_History"
SAVED_MESSAGES_DIR = "savedMessages"
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs("OurMemes", exist_ok=True)
os.makedirs("Saves", exist_ok=True)
os.makedirs(SAVED_MESSAGES_DIR, exist_ok=True)
user_preferences = {}


# Bot Lifecycle Events
@bot.event
# Handles bot startup and announcement
async def on_ready():
    # Load Opus from known working path
    opus_path = "/usr/lib/x86_64-linux-gnu/libopus.so"
    try:
        if not discord.opus.is_loaded():
            discord.opus.load_opus(opus_path)
        logger.info(f"Opus loaded successfully from {opus_path}")
    except Exception as e:
        logger.error(f"Failed to load Opus from {opus_path}: {e}")
        logger.warning(
            "Voice features (.play) will be disabled due to Opus loading failure."
        )
    else:
        logger.info("Voice features enabled with Opus.")

    logger.info(f"Bot is online as {bot.user.name}")
    channel = bot.get_channel(1345184113623040051)
    if channel:
        version = "69.420.25"
        embed = discord.Embed(
            title="Quargle is online",
            description=f"{version} is now live",
            color=discord.Color.red(),
        )
        await channel.send(embed=embed, delete_after=5)
    else:
        logger.error("Channel 1345184113623040051 not found")


@bot.event
# Sets up bot resources on startup
async def setup_hook():
    bot.http_session = aiohttp.ClientSession()
    bot.executor = executor
    bot.sources = await preload_sources()
    bot.memeSources = bot.sources["memeSources"]


# Closes bot resources on shutdown
async def close():
    if hasattr(bot, "http_session") and not bot.http_session.closed:
        await bot.http_session.close()
    bot.executor.shutdown(wait=False)


bot.on_close = close


# File and Data Management Functions
# Loads lines from a file asynchronously
async def load_file(filename):
    try:
        async with aiofiles.open(filename, "r", encoding="utf-8") as file:
            return [line.strip() async for line in file if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return []


# Preloads meme sources from file
async def preload_sources():
    return {
        "memeSources": await load_file("memeSources.txt"),
    }


# Returns path to user’s conversation history file
def get_history_file(user_id):
    return os.path.join(HISTORY_DIR, f"user_{user_id}.txt")


# Returns path to user’s saved messages file
def get_saved_messages_file(user_id):
    return os.path.join(SAVED_MESSAGES_DIR, f"user_{user_id}.json")


# Loads recent conversation history for a user
async def load_conversation_history(user_id):
    file_path = get_history_file(user_id)
    if not os.path.exists(file_path):
        return []
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        lines = [line async for line in f]
    history = []
    for line in lines[-20:]:
        if line.strip():
            try:
                message = json.loads(line)
                history.append(message)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in history file: {line}")
    return history


# Appends a message to user’s conversation history
async def append_to_conversation_history(user_id, role, content):
    file_path = get_history_file(user_id)
    message = {"role": role, "content": content}
    async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
        await f.write(f"{json.dumps(message)}\n")


# Writes a system message to user’s history file
async def write_system_message(user_id, content):
    file_path = get_history_file(user_id)
    system_msg = {"role": "system", "content": content}
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(f"{json.dumps(system_msg)}\n")


# Utility Functions
# Checks if user has a specific permission
async def check_permissions(ctx, permission):
    if not getattr(ctx.author.guild_permissions, permission, False):
        await ctx.send("You lack permission!", delete_after=2)
        return False
    return True


# Cleans up messages after a delay
async def cleanup_messages(command_msg, preview_msg):
    await asyncio.sleep(30)
    try:
        await command_msg.delete()
        await preview_msg.delete()
    except Exception as e:
        logger.debug(f"Failed to delete preview messages: {e}")


# Media Handling Functions
# Saves an attachment or URL to a directory
async def save_attachment(item, session, directory):
    async with session.get(item.url) as resp:
        if resp.status == 200:
            filename = os.path.join(directory, item.filename)
            async with aiofiles.open(filename, "wb") as f:
                await f.write(await resp.read())


# Utility Commands
@bot.command()
@commands.has_permissions(manage_messages=True)
# Deletes a specified number of messages
async def clear(ctx, amount: int):
    if amount > 200:
        await ctx.send("I WON'T DELETE MORE THAN 200 MESSAGES!!!!", delete_after=2)
        return
    await ctx.send(f"Deleting {amount} messages...", delete_after=2)
    deleted = await ctx.channel.purge(limit=amount, bulk=True)
    await ctx.send(f"Deleted {len(deleted)} messages.", delete_after=2)


@clear.error
# Handles permission errors for clear command
async def clear_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need Manage Messages permission!", delete_after=2)


@bot.command()
# Sends a debug message
async def debug(ctx):
    await ctx.send("Debug", delete_after=5)


@bot.command()
# Displays a user’s profile picture
async def getpfp(ctx, member: Member = None):
    member = member or ctx.author
    embed = Embed(title=str(member), url=member.display_avatar.url)
    embed.set_image(url=member.display_avatar.url)
    await ctx.send(embed=embed)


@bot.command()
# Fetches and displays weather forecast for a city
async def weather(ctx, *, city=""):
    if not city:
        await ctx.send("City is missing", delete_after=1)
        return
    async with python_weather.Client(unit=python_weather.IMPERIAL) as wc:
        try:
            weather = await wc.get(city)
            current_temp = weather.temperature
            forecast_msg = [f"Current temperature in {city}: {current_temp}°F"]
            forecast_msg.extend(
                f"{daily.date.strftime('%m/%d')}: High: {daily.highest_temperature}°F, "
                f"Low: {daily.lowest_temperature}°F, Sunset: {daily.sunset.strftime('%I:%M %p')}"
                for daily in weather.daily_forecasts[:3]
            )
            await ctx.send("\n".join(forecast_msg))
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
            await ctx.send("Failed to fetch weather data.", delete_after=2)


# Memes and Fun Commands
@bot.command()
# Sends a freaky message to a specific channel
async def freak(ctx):
    await ctx.message.delete(delay=1)
    target = (
        ctx.message.reference
        and (await ctx.channel.fetch_message(ctx.message.reference.message_id)).author
    )
    mention = target.mention if target else "nobody in particular"
    channel = bot.get_channel(656690392049385484)
    if channel:
        embed = discord.Embed(
            title="😈freak mode activated😈",
            description=f"Im gonna touch you, {mention}",
            color=discord.Color.red(),
        )
        embed.set_image(url="https://c.tenor.com/-A4nRXhIdSEAAAAd/tenor.gif")
        await channel.send(embed=embed, delete_after=45)


@bot.command()
# Fetches and posts a random Reddit meme
async def meme(ctx):
    await ctx.message.delete(delay=1)
    embed = Embed()
    async with bot.http_session as session:
        for meme_url in random.sample(bot.memeSources, len(bot.memeSources)):
            try:
                async with session.get(
                    meme_url, headers={"User-Agent": "meme-bot"}
                ) as r:
                    if r.status != 200:
                        continue
                    res = await r.json()
                    memes = [
                        post["data"]
                        for post in res["data"]["children"]
                        if "url" in post["data"]
                        and not post["data"].get("is_video")
                        and post["data"].get("is_reddit_media_domain")
                        and not post["data"].get("over_18")
                    ]
                    if not memes:
                        continue
                    meme_data = random.choice(memes)
                    embed.title = meme_data["title"][:256]
                    embed.set_image(url=meme_data["url"])
                    await ctx.send(embed=embed)
                    return
            except Exception as e:
                logger.error(f"Meme fetch failed for {meme_url}: {e}")
        await ctx.send("Failed to fetch meme. Try again later!", delete_after=3)


@bot.command()
# Replies with a GIF based on a referenced message
async def reaction(ctx):
    await ctx.message.delete(delay=1)
    if not ctx.message.reference:
        await ctx.send("No message was referenced.", delete_after=2)
        return
    ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
    original_message = ref_msg.content
    username = ref_msg.author.name
    embed = Embed()
    replacements = {"cunt": "jerk", "nazi": "creep", "retard": "goof"}
    words = original_message.split()
    sanitized_message = [
        (
            replacements.get(word.lower(), word).capitalize()
            if word.lower() in replacements
            else word
        )
        for word in words
    ]
    sanitized_message = " ".join(sanitized_message)
    search_term = urllib.parse.quote(sanitized_message)
    tenor_url = f"https://tenor.com/search/{search_term}-gifs"
    async with bot.http_session as session:
        async with session.get(tenor_url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                gif_img = soup.find("img", src=lambda x: x and ".gif" in x)
                if gif_img and gif_img["src"]:
                    embed.title = f"{username}: {original_message}"
                    embed.set_image(url=gif_img["src"])
                    await ref_msg.reply(embed=embed)
                else:
                    await ctx.send("No GIFs found.", delete_after=2)
            else:
                await ctx.send("Failed to load Tenor page.", delete_after=2)


@bot.command()
# Uploads attachments or GIF URLs to specified directory
async def upload(ctx, directory="OurMemes"):
    valid_dirs = ["OurMemes", "Saves"]
    if directory not in valid_dirs:
        await ctx.send(
            f"Invalid directory! Use: {', '.join(valid_dirs)}", delete_after=4
        )
        return
    command_attachments = ctx.message.attachments
    ref_urls = []
    if ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        ref_urls = [
            word for word in ref_msg.content.split() if word.lower().endswith(".gif")
        ]
    all_items = command_attachments + [
        type("obj", (), {"url": url, "filename": url.split("/")[-1]})()
        for url in ref_urls
    ]
    if not all_items:
        await ctx.send("No attachments or GIF links found to upload!", delete_after=4)
        return
    async with aiohttp.ClientSession() as session:
        tasks = [save_attachment(item, session, directory) for item in all_items]
        await asyncio.gather(*tasks)
    num_files = len(tasks)
    if num_files == 1:
        await ctx.send(f"1 file uploaded to {directory}", delete_after=10)
    else:
        await ctx.send(f"{num_files} files uploaded to {directory}", delete_after=10)


@bot.command()
# Shares a random meme from OurMemes directory
async def ourmeme(ctx, media_type: str = None):
    valid_exts = {"image": (".png", ".jpg", ".gif"), "video": (".mp4", ".mov", ".mkv")}
    exts = valid_exts.get(
        media_type.lower() if media_type else None,
        valid_exts["image"] + valid_exts["video"],
    )
    files = [f for f in os.listdir("OurMemes") if f.lower().endswith(exts)]
    if not files:
        await ctx.send(f"No {media_type or 'memes'} found!", delete_after=2)
        return
    file_path = os.path.join("OurMemes", random.choice(files))
    titles = await load_file("Oldwordlist.txt")
    title = random.choice(titles) if titles else "Random Meme"
    file = File(file_path)
    if file_path.lower().endswith(valid_exts["image"]):
        embed = Embed(title=title, color=discord.Color.blue())
        embed.set_image(url=f"attachment://{os.path.basename(file_path)}")
        await ctx.send(embed=embed, file=file)
    else:
        await ctx.send(content=title, file=file)
    await ctx.message.delete(delay=1)


# ASCII characters from dark to light
ASCII_CHARS = "@%#*+=-:. "


def image_to_ascii(image, width=100):
    aspect_ratio = image.height / image.width
    new_height = int(width * aspect_ratio * 0.55)
    image = image.resize((width, new_height)).convert("L")

    ascii_str = "".join(ASCII_CHARS[pixel // 32] for pixel in image.getdata())
    ascii_str = "\n".join(
        ascii_str[i : i + width] for i in range(0, len(ascii_str), width)
    )

    return ascii_str


async def ascii(ctx):
    image = None
    if ctx.message.attachments:
        image_bytes = await ctx.message.attachments[0].read()
        image = Image.open(io.BytesIO(image_bytes))
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if ref_msg.attachments:
            image_bytes = await ref_msg.attachments[0].read()
            image = Image.open(io.BytesIO(image_bytes))

    if image is None:
        await ctx.send("Please upload or reply to an image.")
        return

    ascii_art = image_to_ascii(image)
    file = discord.File(io.BytesIO(ascii_art.encode()), filename="ascii_art.txt")
    await ctx.send("Here is your ASCII art:", file=file)


@bot.command()
# Adds captions to an image from a referenced message or attachment
async def caption(ctx, top_text: str = "", bottom_text: str = ""):
    image_url = None
    if ctx.message.attachments:
        image_url = ctx.message.attachments[0].url
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if ref_msg.attachments:
            image_url = ref_msg.attachments[0].url
        else:
            urls = [
                word
                for word in ref_msg.content.split()
                if word.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
            ]
            if urls:
                image_url = urls[0]
    if not image_url:
        await ctx.send("Please attach an image or reply to one!", delete_after=4)
        return

    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status != 200:
                await ctx.send("Failed to fetch image!", delete_after=4)
                return
            image_data = await resp.read()

    image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = max(20, width // 10)  # Scale font size to 1/10th of image width, min 20
    try:
        font = ImageFont.truetype("comic.ttf", font_size)
    except:
        font = ImageFont.load_default(size=font_size)

    if top_text:
        top_text = top_text.upper()
        top_bbox = draw.textbbox((0, 0), top_text, font=font)
        top_x = (width - (top_bbox[2] - top_bbox[0])) // 2
        draw.text(
            (top_x, 10),
            top_text,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    if bottom_text:
        bottom_text = bottom_text.upper()
        bottom_bbox = draw.textbbox((0, 0), bottom_text, font=font)
        bottom_x = (width - (bottom_bbox[2] - bottom_bbox[0])) // 2
        bottom_y = (
            height - (bottom_bbox[3] - bottom_bbox[1]) - 20
        )  # Added 10px buffer (total 20px from edge)
        draw.text(
            (bottom_x, bottom_y),
            bottom_text,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    if not top_text and not bottom_text:
        await ctx.send(
            "Please provide at least one caption (top or bottom)!", delete_after=4
        )
        return

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    await ctx.send(file=File(buffer, "captioned.png"))


@bot.command()
# Plays a sound effect in the user’s voice channel
async def play(ctx, sound: str):
    sound_files = {
        "laugh": "laugh.mp3",
        "clap": "clap.mp3",
    }
    if sound not in sound_files:
        await ctx.send(
            f"Available sounds: {', '.join(sound_files.keys())}", delete_after=4
        )
        return
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("You need to be in a voice channel!", delete_after=4)
        return
    if not discord.opus.is_loaded():
        await ctx.send(
            "Voice support is not available due to missing Opus library!",
            delete_after=4,
        )
        return
    voice_channel = ctx.author.voice.channel
    vc = await voice_channel.connect()
    sound_path = os.path.join("sounds", sound_files[sound])
    if not os.path.exists(sound_path):
        await ctx.send("Sound file not found!", delete_after=4)
        await vc.disconnect()
        return
    vc.play(discord.FFmpegPCMAudio(sound_path))
    while vc.is_playing():
        await asyncio.sleep(1)
    await vc.disconnect()


# AI Feature Commands
@bot.command()
# Sets custom context for AI responses
async def setcontext(ctx, *, new_context: str):
    user_id = ctx.author.id
    user_preferences[user_id] = new_context
    logger.debug(f"Set context for user {user_id} to: {new_context}")
    await ctx.send(f"Context updated: {new_context}", delete_after=5)


@bot.command()
# Chats with QUARGLE AI using OpenAI
async def QUARGLE(ctx, *, inputText: str):
    openai.api_key = OPENAI_GPT_TOKEN
    user_id = ctx.author.id
    logger.debug(
        f"Processing QUARGLE command for user {user_id} with input: {inputText}"
    )
    sanitized_input = (
        profanity.censor(inputText)
        if profanity.contains_profanity(inputText)
        else inputText
    )
    original_message = ""
    original_author = ""
    if ctx.message.reference:
        try:
            ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            original_message = ref_msg.content
            original_author = ref_msg.author.name
        except Exception as e:
            logger.error(f"Failed to fetch referenced message: {e}")
    username = ctx.author.name
    context = user_preferences.get(user_id, "")
    file_path = get_history_file(user_id)
    conversation_history = await load_conversation_history(user_id)
    if not os.path.exists(file_path) or not conversation_history:
        system_msg = {
            "role": "system",
            "content": f"{BOT_IDENTITY} Assisting {username}. {context}",
        }
        await write_system_message(user_id, system_msg["content"])
        conversation_history = [system_msg]
    conversation_input = sanitized_input
    if original_message:
        conversation_input = (
            f"{sanitized_input}\n\nReplying to {original_author}: '{original_message}'"
        )
    await append_to_conversation_history(user_id, "user", conversation_input)
    conversation_history.append({"role": "user", "content": conversation_input})
    system_msg = {
        "role": "system",
        "content": f"{BOT_IDENTITY} Assisting {username}. {context}",
    }
    api_history = [system_msg] + conversation_history[-20:]
    thinking_message = await ctx.send("Thinking...")
    try:
        response = await bot.loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=api_history,
            ),
        )
        bot_response = response.choices[0].message.content
        await append_to_conversation_history(user_id, "assistant", bot_response)
        await thinking_message.delete()
        await ctx.send(bot_response)
    except Exception as e:
        logger.error(f"QUARGLE error: {e}")
        await ctx.send("An error occurred with the AI.", delete_after=10)
    finally:
        try:
            await thinking_message.delete()
        except:
            pass


@bot.command()
# Generates an image using DALL-E 3
async def imagine(ctx, *, inputText: str):
    loading_msg = await ctx.send("Processing...", delete_after=1)
    try:
        response = await bot.loop.run_in_executor(
            None,
            lambda: openai.Image.create(
                prompt=inputText,
                model="dall-e-3",
                size="512x512",
                n=1,
                response_format="url",
            ),
        )
        embed = Embed(title=inputText, url=response.data[0].url)
        embed.set_image(url=response.data[0].url)
        await ctx.send(embed=embed)
        logger.info(f"Generated image for '{inputText}': {response.data[0].url}")
    except Exception as e:
        logger.error(f"Imagine error: {e}")
        await ctx.send("Failed to generate image.", delete_after=2)


@bot.command()
# Analyzes sentiment of a referenced message
async def sentiment(ctx):
    if not ctx.message.reference:
        await ctx.send("Please reply to a message to analyze!", delete_after=4)
        return
    ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
    openai.api_key = OPENAI_GPT_TOKEN
    prompt = f"Analyze the sentiment of this text: '{ref_msg.content}'"
    try:
        response = await bot.loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        sentiment = response.choices[0].message.content
        embed = Embed(
            title=f"Sentiment Analysis for {ref_msg.author.name}",
            description=sentiment,
            color=discord.Color.blue(),
        )
        embed.add_field(
            name="Original Message", value=ref_msg.content[:1024], inline=False
        )
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        await ctx.send("Failed to analyze sentiment.", delete_after=4)


# Admin Commands
@bot.command()
# Shuts down bot for updates
async def update(ctx):
    await ctx.send("Bot is prepping for updates...", delete_after=1)
    await asyncio.sleep(2)
    await bot.close()


@bot.command()
@commands.has_permissions(administrator=True)
# Clears all conversation history files
async def clearhistory(ctx):
    logger.info(
        f"Clearhistory command invoked by {ctx.author.name} (ID: {ctx.author.id})"
    )
    history_dir = HISTORY_DIR
    if not os.path.exists(history_dir):
        await ctx.send("No conversation history directory found!", delete_after=5)
        logger.warning(f"Directory {history_dir} does not exist")
        return
    files_deleted = 0
    try:
        for filename in os.listdir(history_dir):
            file_path = os.path.join(history_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_deleted += 1
                logger.debug(f"Deleted file: {file_path}")
        if files_deleted > 0:
            await ctx.send(
                f"Cleared {files_deleted} conversation history file(s)!", delete_after=5
            )
            logger.info(
                f"Successfully deleted {files_deleted} files from {history_dir}"
            )
        else:
            await ctx.send("No conversation history files to clear!", delete_after=5)
            logger.info(f"No files found in {history_dir} to delete")
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        await ctx.send(
            "An error occurred while clearing the conversation history.", delete_after=5
        )


@clearhistory.error
# Handles permission errors for clearhistory command
async def clearhistory_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send(
            "You need Administrator permissions to use this command!", delete_after=5
        )
        logger.warning(
            f"{ctx.author.name} (ID: {ctx.author.id}) attempted clearhistory without admin perms"
        )


# Message Management Commands
@bot.command()
# Saves a referenced message to a JSON file
async def savemessage(ctx):
    if not ctx.message.reference:
        await ctx.send("Please reply to a message to save it!", delete_after=5)
        return
    ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
    user_id = ref_msg.author.id
    username = ref_msg.author.name
    content = ref_msg.content
    timestamp = ref_msg.created_at.isoformat()
    file_path = get_saved_messages_file(user_id)
    messages = []
    if os.path.exists(file_path):
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            try:
                messages = json.loads(await f.read())
            except json.JSONDecodeError:
                logger.error(f"Corrupted JSON file for user {user_id}, resetting.")
                messages = []
    messages.append({"content": content, "timestamp": timestamp})
    if len(messages) > 20:
        messages = messages[-20:]
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(messages, indent=2))
    await ctx.send(f"Saved message from {username}!", delete_after=5)
    await ctx.message.delete(delay=1)


class MessageSelect(ui.Select):
    def __init__(self, messages, member):
        self.messages = messages
        self.member = member
        options = [
            SelectOption(
                label=f"Message {i+1}", value=str(i), description=msg["content"][:50]
            )
            for i, msg in enumerate(messages[:25])
        ]
        super().__init__(placeholder="Select a message...", options=options)

    async def callback(self, interaction: discord.Interaction):
        if interaction.user != self.view.ctx.author:
            await interaction.response.send_message(
                "This isn’t your selection!", ephemeral=True
            )
            return
        selected_idx = int(self.values[0])
        msg = self.messages[selected_idx]
        embed = Embed(
            title=f"Message from {self.member.name}",
            description=msg["content"],
            color=discord.Color.gold(),
        )
        embed.set_footer(text=f"Saved on: {msg['timestamp']}")
        await interaction.response.send_message(embed=embed)
        self.view.stop()


class MessageView(ui.View):
    def __init__(self, ctx, messages, member):
        super().__init__(timeout=30)
        self.ctx = ctx
        self.add_item(MessageSelect(messages, member))


@bot.command()
# Lists or retrieves saved messages for a user with pagination
async def mentionmessage(ctx, member: Member, page: int = 1):
    user_id = member.id
    file_path = get_saved_messages_file(user_id)
    if not os.path.exists(file_path):
        await ctx.send(f"No saved messages found for {member.name}!", delete_after=5)
        return
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        try:
            messages = json.loads(await f.read())
        except json.JSONDecodeError:
            logger.error(f"Corrupted JSON file for user {user_id}.")
            await ctx.send("Error reading saved messages!", delete_after=5)
            return
    if not messages:
        await ctx.send(f"No saved messages found for {member.name}!", delete_after=5)
        return
    ITEMS_PER_PAGE = 5
    total_pages = (len(messages) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    if page < 1 or page > total_pages:
        await ctx.send(f"Invalid page! Use 1 to {total_pages}.", delete_after=5)
        return
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    embed = Embed(
        title=f"Saved Messages for {member.name} (Page {page}/{total_pages})",
        color=discord.Color.gold(),
        description="Select a message below or use `.mentionmessage @user <page>`",
    )
    for i, msg in enumerate(messages[start:end], start + 1):
        preview = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
        embed.add_field(name=f"{i}. {msg['timestamp']}", value=preview, inline=False)
    view = MessageView(ctx, messages, member)
    preview_msg = await ctx.send(embed=embed, view=view)
    await view.wait()
    await preview_msg.edit(view=None)


# Help Menu
COMMAND_CATEGORIES = {
    "Utilities": {
        "clear": "Clears up to 200 messages (Manage Messages required)",
        "getpfp": "Shows a user’s avatar (defaults to caller)",
        "weather": "Shows 3-day forecast for a city",
        "debug": "Sends a debug message",
    },
    "Memes & Fun": {
        "freak": "Sends a freaky message to a channel",
        "meme": "Posts a random Reddit meme",
        "reaction": "Replies with a GIF to a referenced message",
        "ourmeme": "Shares a random local meme (image/video)",
        "upload": "Uploads attachments to OurMemes or Saves (e.g., .upload Saves)",
        "caption": "Adds top and bottom text to an image",
        "play": "Plays a sound effect in your voice channel",
    },
    "AI Features": {
        "setcontext": "Sets custom context for AI responses",
        "QUARGLE": "Chats with QUARGLE AI",
        "imagine": "Generates an image with DALL-E 3",
        "sentiment": "Analyzes the sentiment of a referenced message",
    },
    "Admin Tools": {
        "clearhistory": "Clears all conversation history files (Admin required)",
        "update": "Shuts down bot for updates",
    },
    "Message Management": {
        "savemessage": "Saves a replied-to message to a JSON file",
        "mentionmessage": "Lists saved messages with pagination or retrieves one",
    },
}
COLORS = {
    "Utilities": discord.Color.blue(),
    "Memes & Fun": discord.Color.green(),
    "AI Features": discord.Color.purple(),
    "Admin Tools": discord.Color.red(),
    "Message Management": discord.Color.orange(),
}


class HelpSelect(ui.Select):
    def __init__(self):
        options = [
            SelectOption(label=cat, value=cat) for cat in COMMAND_CATEGORIES.keys()
        ]
        super().__init__(placeholder="Select a category...", options=options)

    async def callback(self, interaction: discord.Interaction):
        if interaction.user != self.view.ctx.author:
            await interaction.response.send_message(
                "This isn’t your help menu!", ephemeral=True
            )
            return
        cat = self.values[0]
        embed = Embed(
            title=f"QUARGLE-HELP - {cat}",
            color=COLORS.get(cat, discord.Color.blue()),
            description=f"Commands for {cat.lower()}.",
        )
        for cmd, desc in COMMAND_CATEGORIES[cat].items():
            embed.add_field(name=f".{cmd}", value=desc, inline=False)
        embed.set_footer(text="Prefix: .")
        await interaction.response.edit_message(embed=embed)


class HelpView(ui.View):
    def __init__(self, ctx):
        super().__init__(timeout=60)
        self.ctx = ctx
        self.add_item(HelpSelect())


@bot.command(name="help")
# Displays an interactive help menu with select dropdown
async def help_command(ctx):
    embed = Embed(
        title="QUARGLE-HELP",
        description="Select a category below to view commands.",
        color=discord.Color.blue(),
    )
    embed.set_footer(text="Prefix: .")
    view = HelpView(ctx)
    msg = await ctx.send(embed=embed, view=view)
    await view.wait()
    await msg.edit(view=None)


# Start Bot
if __name__ == "__main__":
    bot.run(BOT_TOKEN)
