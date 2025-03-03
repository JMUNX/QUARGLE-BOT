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
import cv2
import numpy as np

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
HISTORY_DIR = "conversationHistory"
SAVED_MESSAGES_DIR = "savedMessages"
EMOJI_FOLDER = "emojisFolder"
SAVES_FOLDER = "savesFolder"
OURMEMES_FOLDER = "ourMemes"
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(OURMEMES_FOLDER, exist_ok=True)
os.makedirs(SAVES_FOLDER, exist_ok=True)
os.makedirs(EMOJI_FOLDER, exist_ok=True)
os.makedirs(SAVED_MESSAGES_DIR, exist_ok=True)
user_preferences = {}


# Bot Lifecycle Events
@bot.event
async def on_ready():
    opus_path = "/usr/lib/x86_64-linux-gnu/libopus.so"
    try:
        if not discord.opus.is_loaded():
            discord.opus.load_opus(opus_path)
        logger.info(f"Opus loaded successfully from {opus_path}")
    except Exception as e:
        logger.error(f"Failed to load Opus: {e}")
        logger.warning("Voice features disabled.")
    else:
        logger.info("Voice features enabled.")

    logger.info(f"Bot is online as {bot.user.name}")
    channel = bot.get_channel(1345184113623040051)
    if channel:
        version = "69.420.30"
        embed = Embed(
            title="Quargle is online",
            description=f"{version} is now live",
            color=discord.Color.red(),
        )
        await channel.send(embed=embed, delete_after=5)


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


# File and Data Management Functions
async def load_file(filename):
    try:
        async with aiofiles.open(filename, "r", encoding="utf-8") as file:
            return [line.strip() async for line in file if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return []


async def preload_sources():
    return {"memeSources": await load_file("memeSources.txt")}


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
    history = [json.loads(line) for line in lines[-20:] if line.strip()]
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


# Utility Functions
async def check_permissions(ctx, permission):
    if not getattr(ctx.author.guild_permissions, permission, False):
        await ctx.send("You lack permission!", delete_after=2)
        return False
    return True


async def cleanup_messages(command_msg, preview_msg):
    await asyncio.sleep(30)
    try:
        await command_msg.delete()
        await preview_msg.delete()
    except Exception as e:
        logger.debug(f"Failed to delete messages: {e}")


async def save_attachment(item, session, directory):
    async with session.get(item.url) as resp:
        if resp.status == 200:
            filename = os.path.join(directory, item.filename)
            async with aiofiles.open(filename, "wb") as f:
                await f.write(await resp.read())


# Utilities Commands
@bot.command()
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount: int):
    if amount > 200:
        await ctx.send("I WON’T DELETE MORE THAN 200 MESSAGES!!!!", delete_after=2)
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
async def getpfp(ctx, member: Member = None):
    member = member or ctx.author
    embed = Embed(title=str(member), url=member.display_avatar.url)
    embed.set_image(url=member.display_avatar.url)
    await ctx.send(embed=embed)


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
        await ctx.send("Failed to fetch meme!", delete_after=3)


@bot.command()
async def reaction(ctx):
    await ctx.message.delete(delay=1)
    if not ctx.message.reference:
        await ctx.send("No message referenced.", delete_after=2)
        return
    ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
    embed = Embed()
    replacements = {"cunt": "jerk", "nazi": "creep", "retard": "goof"}
    words = ref_msg.content.split()
    sanitized_message = [
        (
            replacements.get(word.lower(), word).capitalize()
            if word.lower() in replacements
            else word
        )
        for word in words
    ]
    search_term = urllib.parse.quote(" ".join(sanitized_message))
    tenor_url = f"https://tenor.com/search/{search_term}-gifs"
    async with bot.http_session as session:
        async with session.get(tenor_url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                gif_img = soup.find("img", src=lambda x: x and ".gif" in x)
                if gif_img and gif_img["src"]:
                    embed.title = f"{ref_msg.author.name}: {ref_msg.content}"
                    embed.set_image(url=gif_img["src"])
                    await ref_msg.reply(embed=embed)
                else:
                    await ctx.send("No GIFs found.", delete_after=2)
            else:
                await ctx.send("Failed to load Tenor page.", delete_after=2)


@bot.command()
async def upload(ctx, directory=OURMEMES_FOLDER):
    valid_dirs = [OURMEMES_FOLDER, SAVES_FOLDER, EMOJI_FOLDER]
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
        await ctx.send("No attachments or GIF links found!", delete_after=4)
        return
    async with aiohttp.ClientSession() as session:
        tasks = [save_attachment(item, session, directory) for item in all_items]
        await asyncio.gather(*tasks)
    await ctx.send(f"{len(tasks)} file(s) uploaded to {directory}", delete_after=10)


@bot.command()
async def ourmeme(ctx, media_type: str = None):
    valid_exts = {"image": (".png", ".jpg", ".gif"), "video": (".mp4", ".mov", ".mkv")}
    exts = valid_exts.get(
        media_type.lower() if media_type else None,
        valid_exts["image"] + valid_exts["video"],
    )
    files = [f for f in os.listdir(OURMEMES_FOLDER) if f.lower().endswith(exts)]
    if not files:
        await ctx.send(f"No {media_type or 'memes'} found!", delete_after=2)
        return
    file_path = os.path.join(OURMEMES_FOLDER, random.choice(files))
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


# PIL Functions for Memes & Fun
ASCII_CHARS_DENSE = "@#S%?*+;:,. "


def image_to_ascii(image, width=50, dense=True):
    aspect_ratio = image.height / image.width
    new_height = int(width * aspect_ratio * 0.55)
    image = image.resize((width, new_height)).convert("L")
    ascii_chars = ASCII_CHARS_DENSE
    ascii_str = "".join(
        ascii_chars[pixel * (len(ascii_chars) - 1) // 255] for pixel in image.getdata()
    )
    return "\n".join(ascii_str[i : i + width] for i in range(0, len(ascii_str), width))


def replace_faces_with_emoji(image, emoji_path):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    emoji = Image.open(emoji_path).convert("RGBA")
    for x, y, w, h in faces:
        emoji_resized = emoji.resize((int(w * 1.2), int(h * 1.2)), Image.LANCZOS)
        emoji_x = x + (w - emoji_resized.width) // 2
        emoji_y = y + (h - emoji_resized.height) // 2
        image.paste(emoji_resized, (emoji_x, emoji_y), emoji_resized)
    return image


@bot.command()
async def ascii(ctx):
    image = None
    if ctx.message.attachments:
        image = Image.open(io.BytesIO(await ctx.message.attachments[0].read()))
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if ref_msg.attachments:
            image = Image.open(io.BytesIO(await ref_msg.attachments[0].read()))
    if image is None:
        await ctx.send("Please upload or reply to an image.")
        return
    ascii_art = image_to_ascii(image, width=100, dense=True)
    file = File(io.BytesIO(ascii_art.encode()), filename="ascii_art.txt")
    await ctx.send("Detailed ASCII art:", file=file)


@bot.command()
async def pixelate(ctx, intensity: int = 5):
    if intensity < 1 or intensity > 10:
        await ctx.send("Intensity must be between 1 and 10.")
        return
    image = None
    if ctx.message.attachments:
        image = Image.open(io.BytesIO(await ctx.message.attachments[0].read()))
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if ref_msg.attachments:
            image = Image.open(io.BytesIO(await ref_msg.attachments[0].read()))
    if image is None:
        await ctx.send("Please upload or reply to an image.")
        return
    pixel_size = intensity * 5
    image = image.resize(
        (image.width // pixel_size, image.height // pixel_size), Image.NEAREST
    )
    image = image.resize(
        (image.width * pixel_size, image.height * pixel_size), Image.NEAREST
    )
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    file = File(img_bytes, filename="pixelated.png")
    await ctx.send(f"Pixelated image (Intensity {intensity}):", file=file)


@bot.command()
async def emojify(ctx, emoji_name: str = None):
    if not emoji_name:
        # List all files with valid image extensions in EMOJI_FOLDER
        valid_extensions = (".png", ".jpg", ".jpeg", ".gif")
        emoji_files = [
            f[: f.rfind(".")]  # Strip extension
            for f in os.listdir(EMOJI_FOLDER)
            if os.path.isfile(os.path.join(EMOJI_FOLDER, f))
            and f.lower().endswith(valid_extensions)
        ]
        if not emoji_files:
            await ctx.send(
                f"No emojis found in the `/{EMOJI_FOLDER}/` folder.", delete_after=10
            )
            return

        # Map emoji names to Unicode placeholders (customize as needed)
        emoji_previews = {
            "freak": "<:freak:1345942275175219221>",
            "death": "<:death:1345942262860873740>",
            "creep": "<:creep:1345942246851215410>",
            "chad": "<:chad:1345942227800817755>",
            # Add more mappings or use a default emoji for unmapped names
        }
        default_preview = "🙂"  # Fallback for unmapped emojis

        # Build the description with names and emoji previews
        description = "Use `.emojify <emoji_name>` with one of these:\n\n"
        for emoji in emoji_files:
            preview = emoji_previews.get(emoji.lower(), default_preview)
            description += f"- {preview} `{emoji}`\n"

        embed = Embed(
            title="Available Emojis",
            description=description,
            color=discord.Color.blue(),
        )

        # Check embed size (Discord limit: 6000 total characters, 1024 per description)
        if len(description) > 1024:
            # Split into multiple embeds if too long
            lines = description.split("\n")
            current_desc = "Use `.emojify <emoji_name>` with one of these:\n\n"
            embeds = []
            for line in lines[2:]:  # Skip initial header lines
                if len(current_desc) + len(line) + 1 > 1024:
                    embeds.append(
                        Embed(
                            title="Available Emojis (Continued)",
                            description=current_desc,
                            color=discord.Color.blue(),
                        )
                    )
                    current_desc = "Use `.emojify <emoji_name>` with one of these:\n\n"
                current_desc += line + "\n"
            if current_desc.strip():
                embeds.append(
                    Embed(
                        title="Available Emojis (Continued)",
                        description=current_desc,
                        color=discord.Color.blue(),
                    )
                )
            embeds[0].title = "Available Emojis"  # First embed keeps original title
            for embed in embeds:
                await ctx.send(embed=embed, delete_after=30)
        else:
            await ctx.send(embed=embed, delete_after=30)
        return

    emoji_path = os.path.join(EMOJI_FOLDER, f"{emoji_name}.png")
    if not os.path.exists(emoji_path):
        await ctx.send(f"Emoji `{emoji_name}` not found in `/{EMOJI_FOLDER}/`.")
        return
    image = None
    if ctx.message.attachments:
        image = Image.open(io.BytesIO(await ctx.message.attachments[0].read()))
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if ref_msg.attachments:
            image = Image.open(io.BytesIO(await ref_msg.attachments[0].read()))
    if image is None:
        await ctx.send("Please upload or reply to an image.")
        return
    image = replace_faces_with_emoji(image, emoji_path)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    file = File(img_bytes, filename="emoji_faces.png")
    await ctx.send(f"Image with `{emoji_name}` emoji faces:", file=file)


@bot.command()
async def caption(ctx, top_text: str = "", bottom_text: str = ""):
    image_url = None
    if ctx.message.attachments:
        image_url = ctx.message.attachments[0].url
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        image_url = (
            ref_msg.attachments[0].url
            if ref_msg.attachments
            else next(
                (
                    word
                    for word in ref_msg.content.split()
                    if word.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
                ),
                None,
            )
        )
    if not image_url:
        await ctx.send("Please attach or reply to an image!", delete_after=4)
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status != 200:
                await ctx.send("Failed to fetch image!", delete_after=4)
                return
            image = Image.open(io.BytesIO(await resp.read())).convert("RGBA")

    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Font setup with Comic Sans
    font_path = "comicz.ttf"
    if not os.path.exists(font_path):
        await ctx.send(
            "Comic Sans font file not found! Using default font.", delete_after=4
        )
        font_path = None
    base_font_size = width // 10  # Starting font size
    max_width = width * 0.9  # 90% of image width for text

    def wrap_text(text, font, max_width):
        """Wrap text into multiple lines to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def get_font_size(text, max_width, max_height, base_size):
        """Dynamically adjust font size to fit text within bounds."""
        font_size = base_size
        while font_size > 10:  # Minimum font size
            font = (
                ImageFont.truetype(font_path, font_size)
                if font_path
                else ImageFont.load_default(size=font_size)
            )
            lines = wrap_text(text, font, max_width)
            total_height = (
                len(lines)
                * (
                    draw.textbbox((0, 0), lines[0], font=font)[3]
                    - draw.textbbox((0, 0), lines[0], font=font)[1]
                    + 5
                )
                - 5
            )
            total_width = max(
                draw.textbbox((0, 0), line, font=font)[2]
                - draw.textbbox((0, 0), line, font=font)[0]
                for line in lines
            )
            if total_width <= max_width and total_height <= max_height:
                return font, lines
            font_size -= 2  # Reduce size incrementally
        # Fallback to smallest size if it still doesn’t fit
        font = (
            ImageFont.truetype(font_path, 10)
            if font_path
            else ImageFont.load_default(size=10)
        )
        return font, wrap_text(text, font, max_width)

    if top_text:
        top_text = top_text.upper()
        font, top_lines = get_font_size(
            top_text, max_width, height // 3, base_font_size
        )  # Limit to top third
        line_height = (
            draw.textbbox((0, 0), top_lines[0], font=font)[3]
            - draw.textbbox((0, 0), top_lines[0], font=font)[1]
            + 5
        )
        y_offset = 10  # Starting padding from top
        for line in top_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (width - (bbox[2] - bbox[0])) // 2
            draw.text(
                (x, y_offset),
                line,
                font=font,
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            y_offset += line_height

    if bottom_text:
        bottom_text = bottom_text.upper()
        font, bottom_lines = get_font_size(
            bottom_text, max_width, height // 3, base_font_size
        )  # Limit to bottom third
        line_height = (
            draw.textbbox((0, 0), bottom_lines[0], font=font)[3]
            - draw.textbbox((0, 0), bottom_lines[0], font=font)[1]
            + 5
        )
        total_height = len(bottom_lines) * line_height
        y_offset = height - total_height - 20  # Ensure it fits above bottom edge
        if y_offset < height // 3 * 2:  # Prevent overlap with top text
            y_offset = height // 3 * 2  # Start at 2/3rds height if needed
            font, bottom_lines = get_font_size(
                bottom_text, max_width, height - y_offset - 20, base_font_size
            )  # Recalculate with new height
            line_height = (
                draw.textbbox((0, 0), bottom_lines[0], font=font)[3]
                - draw.textbbox((0, 0), bottom_lines[0], font=font)[1]
                + 5
            )
            total_height = len(bottom_lines) * line_height
            y_offset = height - total_height - 20  # Recalculate y_offset
        for line in bottom_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (width - (bbox[2] - bbox[0])) // 2
            draw.text(
                (x, y_offset),
                line,
                font=font,
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            y_offset += line_height

    if not top_text and not bottom_text:
        await ctx.send("Provide at least one caption!", delete_after=4)
        return

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    await ctx.send(file=File(buffer, "captioned.png"))


@bot.command()
async def play(ctx, sound: str):
    sound_files = {"laugh": "laugh.mp3", "clap": "clap.mp3"}
    if sound not in sound_files:
        await ctx.send(
            f"Available sounds: {', '.join(sound_files.keys())}", delete_after=4
        )
        return
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Join a voice channel first!", delete_after=4)
        return
    if not discord.opus.is_loaded():
        await ctx.send("Voice support unavailable!", delete_after=4)
        return
    vc = await ctx.author.voice.channel.connect()
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
async def setcontext(ctx, *, new_context: str):
    user_id = ctx.author.id
    user_preferences[user_id] = new_context
    await ctx.send(f"Context updated: {new_context}", delete_after=5)


@bot.command()
async def QUARGLE(ctx, *, inputText: str):
    openai.api_key = OPENAI_GPT_TOKEN
    user_id = ctx.author.id
    sanitized_input = (
        profanity.censor(inputText)
        if profanity.contains_profanity(inputText)
        else inputText
    )
    original_message = original_author = ""
    if ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        original_message, original_author = ref_msg.content, ref_msg.author.name
    context = user_preferences.get(user_id, "")
    conversation_history = await load_conversation_history(user_id)
    if not conversation_history:
        system_msg = {
            "role": "system",
            "content": f"{BOT_IDENTITY} Assisting {ctx.author.name}. {context}",
        }
        await write_system_message(user_id, system_msg["content"])
        conversation_history = [system_msg]
    conversation_input = (
        f"{sanitized_input}\n\nReplying to {original_author}: '{original_message}'"
        if original_message
        else sanitized_input
    )
    await append_to_conversation_history(user_id, "user", conversation_input)
    conversation_history.append({"role": "user", "content": conversation_input})
    system_msg = {
        "role": "system",
        "content": f"{BOT_IDENTITY} Assisting {ctx.author.name}. {context}",
    }
    api_history = [system_msg] + conversation_history[-20:]
    thinking_message = await ctx.send("Thinking...")
    try:
        response = await bot.loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model="gpt-4o", messages=api_history
            ),
        )
        bot_response = response.choices[0].message.content
        await append_to_conversation_history(user_id, "assistant", bot_response)
        await thinking_message.delete()
        await ctx.send(bot_response)
    except Exception as e:
        logger.error(f"QUARGLE error: {e}")
        await ctx.send("AI error occurred.", delete_after=10)
    finally:
        try:
            await thinking_message.delete()
        except:
            pass


@bot.command()
async def imagine(ctx, *, inputText: str):
    await ctx.send("Processing...", delete_after=1)
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
    except Exception as e:
        logger.error(f"Imagine error: {e}")
        await ctx.send("Failed to generate image.", delete_after=2)


# Admin Commands
@bot.command()
async def update(ctx):
    await ctx.send("Prepping for updates...", delete_after=1)
    await asyncio.sleep(2)
    await bot.close()


@bot.command()
@commands.has_permissions(administrator=True)
async def clearhistory(ctx):
    if not os.path.exists(HISTORY_DIR):
        await ctx.send("No history directory found!", delete_after=5)
        return
    files_deleted = 0
    for filename in os.listdir(HISTORY_DIR):
        file_path = os.path.join(HISTORY_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            files_deleted += 1
    await ctx.send(
        (
            f"Cleared {files_deleted} history file(s)!"
            if files_deleted
            else "No history files to clear!"
        ),
        delete_after=5,
    )


@clearhistory.error
async def clearhistory_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need Administrator permissions!", delete_after=5)


# Help Menu
COMMAND_CATEGORIES = {
    "Utilities": {
        "clear": "Clears up to 200 messages (Manage Messages required)",
        "getpfp": "Shows a user’s avatar (defaults to caller)",
        "debug": "Sends a debug message",
    },
    "Memes & Fun": {
        "meme": "Posts a random Reddit meme",
        "reaction": "Replies with a GIF to a referenced message",
        "ourmeme": "Shares a random local meme (image/video)",
        "upload": "Uploads attachments to ourMemes, Saves or emojis",
        "ascii": "Converts image to detailed ASCII art",
        "pixelate": "Pixelates an image (intensity 1-10)",
        "emojify": "Replaces faces in image with an emoji",
        "caption": "Adds top/bottom text to an image",
        "play": "Plays a sound effect in voice channel",
    },
    "AI Features": {
        "setcontext": "Sets custom context for AI responses",
        "QUARGLE": "Chats with QUARGLE AI",
        "imagine": "Generates an image with DALL-E 3",
    },
    "Admin Tools": {
        "clearhistory": "Clears all conversation history (Admin required)",
        "update": "Shuts down bot for updates",
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
                "Not your help menu!", ephemeral=True
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
async def help_command(ctx):
    embed = Embed(
        title="QUARGLE-HELP",
        description="Select a category below.",
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
