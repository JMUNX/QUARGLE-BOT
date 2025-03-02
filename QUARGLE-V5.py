import discord
from discord import Member, Embed, File
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("TOKENS.env")
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_GPT_TOKEN = os.getenv("OPENAI_GPT_TOKEN")

if not BOT_TOKEN or not OPENAI_GPT_TOKEN:
    logger.critical("Missing BOT_TOKEN or OPENAI_GPT_TOKEN in TOKENS.env")
    exit(1)

# Bot setup with intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(
    command_prefix=".", intents=intents, case_insensitive=True, max_messages=1000
)
bot.remove_command("help")  # Remove default help

# Global resources
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


# Preload sources
async def load_file(filename):
    try:
        async with aiofiles.open(filename, "r", encoding="utf-8") as file:
            return [line.strip() async for line in file if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return []


async def preload_sources():
    return {
        "memeSources": await load_file("memeSources.txt"),
    }


# Bot events
@bot.event
async def on_ready():
    logger.info(f"Bot is online as {bot.user.name}")
    channel = bot.get_channel(1345184113623040051)
    if channel:
        version = "69.420.23"
        embed = discord.Embed(
            title="Quargle is online",
            description=f"{version} is now live",
            color=discord.Color.red(),
        )
        await channel.send(embed=embed, delete_after=5)
    else:
        logger.error("Channel 1345184113623040051 not found")


@bot.event
async def setup_hook():
    bot.http_session = aiohttp.ClientSession()
    bot.executor = executor
    bot.sources = await preload_sources()
    bot.memeSources = bot.sources["memeSources"]


async def close():
    if hasattr(bot, "http_session") and not bot.http_session.closed:
        await bot.http_session.close()
    bot.executor.shutdown(wait=False)


bot.on_close = close


# Utility functions
async def check_permissions(ctx, permission):
    if not getattr(ctx.author.guild_permissions, permission, False):
        await ctx.send("You lack permission!", delete_after=2)
        return False
    return True


def get_history_file(user_id):
    return os.path.join(HISTORY_DIR, f"user_{user_id}.txt")


def get_saved_messages_file(user_id):
    return os.path.join(SAVED_MESSAGES_DIR, f"user_{user_id}.json")


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


async def append_to_conversation_history(user_id, role, content):
    file_path = get_history_file(user_id)
    message = {"role": role, "content": content}
    async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
        await f.write(f"{json.dumps(message)}\n")


async def write_system_message(user_id, content):
    file_path = get_history_file(user_id)
    system_msg = {"role": "system", "content": content}
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(f"{json.dumps(system_msg)}\n")


# Commands
@bot.command()
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount: int):
    if amount > 200:
        await ctx.send("I WON'T DELETE MORE THAN 200 MESSAGES!!!!", delete_after=2)
        return
    await ctx.send(f"Deleting {amount} messages...", delete_after=2)
    deleted = await ctx.channel.purge(limit=amount, bulk=True)
    await ctx.send(f"Deleted {len(deleted)} messages.", delete_after=2)


@clear.error
async def clear_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need Manage Messages permission!", delete_after=2)


@bot.command()
async def debug(ctx):
    await ctx.send("Debug", delete_after=5)


@bot.command()
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
async def update(ctx):
    await ctx.send("Bot is prepping for updates...", delete_after=1)
    await asyncio.sleep(2)
    await bot.close()


@bot.command()
@commands.has_permissions(administrator=True)
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
async def clearhistory_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send(
            "You need Administrator permissions to use this command!", delete_after=5
        )
        logger.warning(
            f"{ctx.author.name} (ID: {ctx.author.id}) attempted clearhistory without admin perms"
        )


@bot.command()
async def getpfp(ctx, member: Member = None):
    member = member or ctx.author
    embed = Embed(title=str(member), url=member.display_avatar.url)
    embed.set_image(url=member.display_avatar.url)
    await ctx.send(embed=embed)


@bot.command()
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


@bot.command()
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
async def upload(ctx):
    command_attachments = ctx.message.attachments
    ref_urls = []

    # Check if this is a reply and extract GIF URLs from the referenced message
    if ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        ref_urls = [
            word for word in ref_msg.content.split() if word.lower().endswith(".gif")
        ]

    # Combine attachments and URLs into a single list for processing
    all_items = command_attachments + [
        type("obj", (), {"url": url, "filename": url.split("/")[-1]})()
        for url in ref_urls
    ]

    if not all_items:
        await ctx.send("No attachments or GIF links found to upload!", delete_after=4)
        return

    async with aiohttp.ClientSession() as session:
        tasks = [save_attachment(item, session, "OurMemes") for item in all_items]
        await asyncio.gather(*tasks)

    num_files = len(tasks)
    if num_files == 1:
        await ctx.send("1 file uploaded", delete_after=10)
    else:
        await ctx.send(f"{num_files} files uploaded", delete_after=10)


async def save_attachment(item, session, directory):
    async with session.get(item.url) as resp:
        if resp.status == 200:
            filename = os.path.join(directory, item.filename)
            async with aiofiles.open(filename, "wb") as f:
                await f.write(await resp.read())


@bot.command()
async def save(ctx):
    command_attachments = ctx.message.attachments
    ref_urls = []

    # Check if this is a reply and extract GIF URLs from the referenced message
    if ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        ref_urls = [
            word for word in ref_msg.content.split() if word.lower().endswith(".gif")
        ]

    # Combine attachments and URLs into a single list for processing
    all_items = command_attachments + [
        type("obj", (), {"url": url, "filename": url.split("/")[-1]})()
        for url in ref_urls
    ]

    if not all_items:
        await ctx.send("No attachments or GIF links found to upload!", delete_after=4)
        return

    async with aiohttp.ClientSession() as session:
        tasks = [save_attachment(item, session, "Saves") for item in all_items]
        await asyncio.gather(*tasks)

    num_files = len(tasks)
    if num_files == 1:
        await ctx.send("1 file uploaded", delete_after=10)
    else:
        await ctx.send(f"{num_files} files uploaded", delete_after=10)


async def save_attachment(item, session, directory):
    async with session.get(item.url) as resp:
        if resp.status == 200:
            filename = os.path.join(directory, item.filename)
            async with aiofiles.open(filename, "wb") as f:
                await f.write(await resp.read())


@bot.command()
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


@bot.command()
async def view(ctx, directory="OurMemes"):
    # Validate directory
    valid_dirs = ["Saves", "OurMemes"]
    if directory not in valid_dirs:
        await ctx.send(
            f"Invalid directory! Use one of: {', '.join(valid_dirs)}", delete_after=4
        )
        return

    # Get list of image files
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
    files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

    if not files:
        await ctx.send(f"No images found in {directory}!", delete_after=4)
        return

    # Send thumbnails as attachments and create a selection menu
    message_content = "Select an image to send by replying with its number:\n"
    attachments = []
    for i, file in enumerate(files[:10]):  # Limit to 10 files to avoid Discord limits
        file_path = os.path.join(directory, file)
        attachments.append(discord.File(file_path, filename=file))
        message_content += f"{i + 1}. {file}\n"

    preview_msg = await ctx.send(content=message_content, files=attachments)

    # Wait for user response
    def check(m):
        return (
            m.author == ctx.author
            and m.channel == ctx.channel
            and m.reference
            and m.reference.message_id == preview_msg.id
        )

    try:
        response = await bot.wait_for("message", timeout=30.0, check=check)
        choice = int(response.content) - 1
        if 0 <= choice < len(files[:10]):
            selected_file = files[choice]
            file_path = os.path.join(directory, selected_file)
            await ctx.send(file=discord.File(file_path, filename=selected_file))
        else:
            await ctx.send("Invalid selection!", delete_after=4)
    except asyncio.TimeoutError:
        await ctx.send("Selection timed out!", delete_after=4)
    except ValueError:
        await ctx.send("Please reply with a valid number!", delete_after=4)


@bot.command()
async def setcontext(ctx, *, new_context: str):
    user_id = ctx.author.id
    user_preferences[user_id] = new_context
    logger.debug(f"Set context for user {user_id} to: {new_context}")
    await ctx.send(f"Context updated: {new_context}", delete_after=5)


@bot.command()
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

    # Load existing messages if file exists
    if os.path.exists(file_path):
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            try:
                messages = json.loads(await f.read())
            except json.JSONDecodeError:
                logger.error(f"Corrupted JSON file for user {user_id}, resetting.")
                messages = []

    # Append new message
    messages.append({"content": content, "timestamp": timestamp})

    # Limit to 20 messages, removing oldest if necessary
    if len(messages) > 20:
        messages = messages[-20:]

    # Save updated messages
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(messages, indent=2))

    await ctx.send(f"Saved message from {username}!", delete_after=5)
    await ctx.message.delete(delay=1)


@bot.command()
async def mentionmessage(ctx, member: Member, message_number: int = None):
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

    if message_number is None:
        # Show preview of all messages
        embed = Embed(
            title=f"Saved Messages for {member.name}",
            color=discord.Color.gold(),
            description="Select a message by using `.mentionmessage @user <number>`",
        )
        for i, msg in enumerate(messages, 1):
            preview = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
            embed.add_field(
                name=f"{i}. {msg['timestamp']}", value=preview, inline=False
            )
        preview_msg = await ctx.send(embed=embed)

        # Store the preview message ID for later deletion
        bot.loop.create_task(cleanup_messages(ctx.message, preview_msg))
    else:
        # Retrieve specific message
        if not 1 <= message_number <= len(messages):
            await ctx.send(
                f"Invalid message number! Use 1 to {len(messages)}.", delete_after=5
            )
            return

        selected_msg = messages[message_number - 1]
        embed = Embed(
            title=f"Message from {member.name}",
            description=selected_msg["content"],
            color=discord.Color.gold(),
        )
        embed.set_footer(text=f"Saved on: {selected_msg['timestamp']}")
        await ctx.send(embed=embed)
        await ctx.message.delete(delay=1)


async def cleanup_messages(command_msg, preview_msg):
    await asyncio.sleep(30)  # Wait 30 seconds for user to select
    try:
        await command_msg.delete()
        await preview_msg.delete()
    except Exception as e:
        logger.debug(f"Failed to delete preview messages: {e}")


# Help menu
COMMAND_CATEGORIES = {
    "Utilities": {
        "clear": "Clears up to 100 messages (Manage Messages required)",
        "getpfp": "Shows a user’s avatar (defaults to caller)",
        "weather": "Shows 3-day forecast for a city",
        "debug": "Sends a debug message",
    },
    "Memes & Fun": {
        "freak": "Sends a freaky message to a channel",
        "meme": "Posts a random Reddit meme",
        "reaction": "Replies with a GIF to a referenced message",
        "ourmeme": "Shares a random local meme (image/video)",
        "upload": "Uploads attachments to local meme storage",
    },
    "AI Features": {
        "setcontext": "Sets custom context for AI responses",
        "QUARGLE": "Chats with QUARGLE AI",
        "imagine": "Generates an image with DALL-E 3",
    },
    "Admin Tools": {
        "clearhistory": "Clears all conversation history files (Admin required)",
        "update": "Shuts down bot for updates",
    },
    "Message Management": {
        "savemessage": "Saves a replied-to message to a JSON file",
        "mentionmessage": "Lists or retrieves saved messages for a user",
    },
}
COLORS = {
    "Utilities": discord.Color.blue(),
    "Memes & Fun": discord.Color.green(),
    "AI Features": discord.Color.purple(),
    "Admin Tools": discord.Color.red(),
    "Message Management": discord.Color.orange(),
}


@bot.command(name="help")
async def help_command(ctx):
    pages = []
    for i, (cat, cmds) in enumerate(COMMAND_CATEGORIES.items()):
        embed = Embed(
            title=f"QUARGLE-HELP - {cat}",
            color=COLORS.get(cat, discord.Color.blue()),
            description=f"Commands for {cat.lower()}.",
        )
        for cmd, desc in cmds.items():
            embed.add_field(name=f".{cmd}", value=desc, inline=False)
        embed.set_footer(text=f"Page {i+1}/{len(COMMAND_CATEGORIES)} | Prefix: .")
        pages.append(embed)

    current_page = 0
    message = await ctx.send(embed=pages[current_page])
    await message.add_reaction("⬅️")
    await message.add_reaction("➡️")
    await message.add_reaction("❌")

    def check(reaction, user):
        return (
            user == ctx.author
            and reaction.message.id == message.id
            and str(reaction.emoji) in ["⬅️", "➡️", "❌"]
        )

    while True:
        try:
            reaction, user = await bot.wait_for(
                "reaction_add", timeout=60.0, check=check
            )
            if str(reaction.emoji) == "➡️" and current_page < len(pages) - 1:
                current_page += 1
            elif str(reaction.emoji) == "⬅️" and current_page > 0:
                current_page -= 1
            elif str(reaction.emoji) == "❌":
                await message.delete()
                break
            await message.edit(embed=pages[current_page])
            await message.remove_reaction(reaction, user)
        except asyncio.TimeoutError:
            await message.clear_reactions()
            break


# Start bot
if __name__ == "__main__":
    bot.run(BOT_TOKEN)
