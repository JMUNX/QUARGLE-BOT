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
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
import cv2
import numpy as np
from typing import Optional, Dict

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
bot_images: Dict[int, Image.Image] = {}  # Map message IDs to bot images


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
        version = "69.420.42"
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
async def load_file(filename: str) -> list:
    try:
        async with aiofiles.open(filename, "r", encoding="utf-8") as file:
            return [line.strip() async for line in file if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return []


async def preload_sources() -> dict:
    return {"memeSources": await load_file("memeSources.txt")}


def get_history_file(user_id: int) -> str:
    return os.path.join(HISTORY_DIR, f"user_{user_id}.txt")


def get_saved_messages_file(user_id: int) -> str:
    return os.path.join(SAVED_MESSAGES_DIR, f"user_{user_id}.json")


async def load_conversation_history(user_id: int) -> list:
    file_path = get_history_file(user_id)
    if not os.path.exists(file_path):
        return []
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        lines = [line async for line in f]
    return [json.loads(line) for line in lines[-20:] if line.strip()]


async def append_to_conversation_history(user_id: int, role: str, content: str) -> None:
    file_path = get_history_file(user_id)
    message = {"role": role, "content": content}
    async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
        await f.write(f"{json.dumps(message)}\n")


async def write_system_message(user_id: int, content: str) -> None:
    file_path = get_history_file(user_id)
    system_msg = {"role": "system", "content": content}
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(f"{json.dumps(system_msg)}\n")


# Utility Functions
async def check_permissions(ctx, permission: str) -> bool:
    if not getattr(ctx.author.guild_permissions, permission, False):
        await ctx.send("You lack permission!", delete_after=2)
        return False
    return True


async def cleanup_messages(command_msg, preview_msg) -> None:
    await asyncio.sleep(30)
    try:
        await command_msg.delete()
        await preview_msg.delete()
    except Exception as e:
        logger.debug(f"Failed to delete messages: {e}")


async def save_attachment(item, session: aiohttp.ClientSession, directory: str) -> None:
    async with session.get(item.url) as resp:
        if resp.status == 200:
            filename = os.path.join(directory, item.filename)
            async with aiofiles.open(filename, "wb") as f:
                await f.write(await resp.read())
        else:
            logger.warning(f"Failed to fetch attachment from {item.url}: {resp.status}")


async def get_image_from_context(ctx) -> Optional[Image.Image]:
    if ctx.message.attachments:
        return Image.open(io.BytesIO(await ctx.message.attachments[0].read()))
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if ref_msg.attachments:
            return Image.open(io.BytesIO(await ref_msg.attachments[0].read()))
        elif ref_msg.author == bot.user and ref_msg.id in bot_images:
            return bot_images[ref_msg.id]
    return None


async def save_image_data(image: Image.Image, directory: str, filename: str) -> None:
    file_path = os.path.join(directory, filename)
    async with aiofiles.open(file_path, "wb") as f:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        await f.write(img_bytes.getvalue())
    logger.debug(f"Saved image to {file_path}")


# Utility Commands
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


# Meme & Media Commands
@bot.command()
async def meme(ctx):
    await ctx.message.delete(delay=1)
    embed = Embed()
    for meme_url in random.sample(bot.memeSources, len(bot.memeSources)):
        try:
            async with bot.http_session.get(
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
    async with bot.http_session.get(tenor_url) as response:
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
    ref_attachments = []
    bot_image_to_upload = None

    logger.debug(f"Command attachments: {len(command_attachments)}")

    if ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        ref_urls = [
            word
            for word in ref_msg.content.split()
            if word.lower().endswith((".gif", ".png", ".jpg", ".jpeg"))
        ]
        ref_attachments = ref_msg.attachments if ref_msg.author != bot.user else []
        logger.debug(
            f"Ref msg by {ref_msg.author}, attachments: {len(ref_msg.attachments)}, URLs: {ref_urls}"
        )

        if ref_msg.author == bot.user:
            # Check if the referenced bot message has attachments
            if ref_msg.attachments:
                bot_image_to_upload = {
                    "image": bot_images[ref_msg.id],
                    "filename": ref_msg.attachments[0].filename,
                }
                logger.debug(f"Bot image detected: {bot_image_to_upload['filename']}")
                await ctx.send(f"Bot image detected: {bot_image_to_upload['filename']}")
            else:
                logger.debug("Bot message referenced, but no image stored")
                await ctx.send("Bot message referenced, but no image stored")

    all_items = (
        command_attachments
        + ref_attachments
        + [
            type("obj", (), {"url": url, "filename": url.split("/")[-1]})()
            for url in ref_urls
        ]
    )
    logger.debug(f"Regular items to upload: {len(all_items)}")

    uploaded_count = 0
    if all_items:
        tasks = [
            save_attachment(item, bot.http_session, directory) for item in all_items
        ]
        await asyncio.gather(*tasks)
        uploaded_count += len(tasks)
    if bot_image_to_upload:
        await save_image_data(
            bot_image_to_upload["image"], directory, bot_image_to_upload["filename"]
        )
        uploaded_count += 1

    if uploaded_count == 0:
        error_msg = (
            "No attachments or valid links found!"
            if not bot_image_to_upload
            else "Bot image unavailable!"
        )
        logger.debug(error_msg)
        await ctx.send(error_msg, delete_after=4)
        return

    await ctx.send(f"{uploaded_count} file(s) uploaded to {directory}", delete_after=10)


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


# Image Processing Commands
ASCII_CHARS_DENSE = "@#S%?*+;:,. "


def image_to_ascii(image: Image.Image, width: int = 50, dense: bool = True) -> str:
    aspect_ratio = image.height / image.width
    new_height = int(width * aspect_ratio * 0.55)
    image = image.resize((width, new_height)).convert("L")
    ascii_chars = ASCII_CHARS_DENSE
    pixels = np.array(image)
    ascii_str = "".join(
        ascii_chars[pixel * (len(ascii_chars) - 1) // 255] for pixel in pixels.flatten()
    )
    return "\n".join(ascii_str[i : i + width] for i in range(0, len(ascii_str), width))


def replace_faces_with_emoji(image: Image.Image, emoji_path: str) -> Image.Image:
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


def deepfry_image(image: Image.Image, intensity: int) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1 + intensity / 5.0)
    img_array = np.array(image)
    noise = np.random.randint(
        -intensity * 5, intensity * 5 + 1, img_array.shape, dtype=np.int16
    )
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_array)
    return image.filter(ImageFilter.GaussianBlur(radius=intensity / 5.0))


@bot.command()
async def ascii(ctx):
    image = await get_image_from_context(ctx)
    if image is None:
        await ctx.send("Please upload or reply to an image.")
        return
    ascii_art = image_to_ascii(image, width=100)
    file = File(io.BytesIO(ascii_art.encode()), filename="ascii_art.txt")
    await ctx.send("Detailed ASCII art:", file=file)


@bot.command()
async def pixelate(ctx, intensity: int = 5):
    if not 1 <= intensity <= 10:
        await ctx.send("Intensity must be between 1 and 10.")
        return
    image = await get_image_from_context(ctx)
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
async def deepfry(ctx, intensity: int = 5):
    if not 1 <= intensity <= 10:
        await ctx.send("Intensity must be between 1 and 10.")
        return
    image = await get_image_from_context(ctx)
    if image is None:
        await ctx.send("Please upload or reply to an image.")
        return
    image = deepfry_image(image, intensity)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    file = File(img_bytes, filename="deepfried.png")
    await ctx.send(f"Deepfried image (Intensity {intensity}):", file=file)


@bot.command()
async def emojify(ctx, emoji_name: str = None):
    if not emoji_name:
        valid_extensions = (".png", ".jpg", ".jpeg", ".gif")
        emoji_files = [
            f[: f.rfind(".")]
            for f in os.listdir(EMOJI_FOLDER)
            if f.lower().endswith(valid_extensions)
        ]
        if not emoji_files:
            await ctx.send(f"No emojis found in `/{EMOJI_FOLDER}/`.", delete_after=10)
            return
        emoji_previews = {
            "freak": "<:freak:1345942275175219221>",
            "death": "<:death:1345942262860873740>",
            "creep": "<:creep:1345942246851215410>",
            "chad": "<:chad:1345942227800817755>",
        }
        default_preview = "🙂"
        description = "Use `.emojify <emoji_name>` with one of these:\n\n" + "\n".join(
            f"- {emoji_previews.get(emoji.lower(), default_preview)} `{emoji}`"
            for emoji in emoji_files
        )
        embed = Embed(
            title="Available Emojis",
            description=description[:1024],
            color=discord.Color.blue(),
        )
        await ctx.send(embed=embed, delete_after=30)
        return
    emoji_path = os.path.join(EMOJI_FOLDER, f"{emoji_name}.png")
    if not os.path.exists(emoji_path):
        await ctx.send(f"Emoji `{emoji_name}` not found in `/{EMOJI_FOLDER}/`.")
        return
    image = await get_image_from_context(ctx)
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
    async with bot.http_session.get(image_url) as resp:
        if resp.status != 200:
            await ctx.send("Failed to fetch image!", delete_after=4)
            return
        image = Image.open(io.BytesIO(await resp.read())).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_path = "comicz.ttf"
    base_font_size = width // 15
    padding = 30

    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                lines.append(word)
                current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def fit_text(text, max_width, max_height, base_size):
        font_size = base_size
        font = (
            ImageFont.truetype(font_path, font_size)
            if os.path.exists(font_path)
            else ImageFont.load_default(size=font_size)
        )
        lines = wrap_text(text, font, max_width)
        line_height = (
            draw.textbbox((0, 0), lines[0] if lines else "A", font=font)[3]
            - draw.textbbox((0, 0), lines[0] if lines else "A", font=font)[1]
        )
        total_height = len(lines) * (line_height + 5)
        while total_height > max_height and font_size > 10:
            font_size -= 2
            font = (
                ImageFont.truetype(font_path, font_size)
                if os.path.exists(font_path)
                else ImageFont.load_default(size=font_size)
            )
            lines = wrap_text(text, font, max_width)
            line_height = (
                draw.textbbox((0, 0), lines[0] if lines else "A", font=font)[3]
                - draw.textbbox((0, 0), lines[0] if lines else "A", font=font)[1]
            )
            total_height = len(lines) * (line_height + 5)
        return font, lines, line_height

    region_height = height // 3 - padding
    if top_text:
        top_text = top_text.upper()
        font, top_lines, line_height = fit_text(
            top_text, width, region_height, base_font_size
        )
        y_offset = padding
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
            y_offset += line_height + 5
    if bottom_text:
        bottom_text = bottom_text.upper()
        font, bottom_lines, line_height = fit_text(
            bottom_text, width, region_height, base_font_size
        )
        total_height = len(bottom_lines) * (line_height + 5)
        y_offset = height - total_height - padding
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
            y_offset += line_height + 5
    if not top_text and not bottom_text:
        await ctx.send("Provide at least one caption!", delete_after=4)
        return
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    await ctx.send(file=File(buffer, "captioned.png"))


# Voice Commands
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
    user_preferences[ctx.author.id] = new_context
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


# Event Handlers
@bot.event
async def on_message(message):
    if message.author == bot.user and message.attachments:
        try:
            bot_images[message.id] = Image.open(
                io.BytesIO(await message.attachments[0].read())
            )
            logger.debug(
                f"Stored bot image from message {message.id}: {message.attachments[0].filename}"
            )
            # Optional: Clean up old images to manage memory
            if len(bot_images) > 10:
                oldest_id = min(bot_images.keys())
                del bot_images[oldest_id]
                logger.debug(f"Removed old bot image {oldest_id} to manage memory")
        except Exception as e:
            logger.error(f"Failed to store bot image from message {message.id}: {e}")
    await bot.process_commands(message)


# Help Menu
COMMAND_CATEGORIES = {
    "Utilities": {
        "clear": "Clears up to 200 messages (Manage Messages required)",
        "getpfp": "Shows a user’s avatar (defaults to caller)",
        "debug": "Sends a debug message",
    },
    "Meme & Media": {
        "meme": "Posts a random Reddit meme",
        "reaction": "Replies with a GIF to a referenced message",
        "ourmeme": "Shares a random local meme (image/video)",
        "upload": "Uploads attachments or bot images to specified folder",
    },
    "Image Processing": {
        "ascii": "Converts image to detailed ASCII art",
        "pixelate": "Pixelates an image (intensity 1-10)",
        "deepfry": "Deepfries an image (intensity 1-10)",
        "emojify": "Replaces faces in image with an emoji",
        "caption": "Adds top/bottom text to an image",
    },
    "Voice": {
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
    "Meme & Media": discord.Color.green(),
    "Image Processing": discord.Color.orange(),
    "Voice": discord.Color.gold(),
    "AI Features": discord.Color.purple(),
    "Admin Tools": discord.Color.red(),
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
