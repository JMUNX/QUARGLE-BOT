import discord
from discord import Member, FFmpegPCMAudio, Embed
from discord.ext import commands, tasks
import random
import asyncio
import aiohttp
import openai
from gtts import gTTS
import os
import requests
import urllib
from urllib.parse import urlparse
import python_weather
from datetime import datetime
from bs4 import BeautifulSoup
import aiofiles
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import logging
import time

# Configure logging for debugging and performance tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version 69.420.1
load_dotenv("TOKENS.env")
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_GPT_TOKEN = os.getenv("OPENAI_GPT_TOKEN")


# notes: Asynchronously loads text files into lists, used for preloading meme and other sources
async def load_file(filename):
    async with aiofiles.open(filename, "r") as file:
        return [line.strip() async for line in file]


# notes: Preloads meme and bham sources at startup to avoid runtime file I/O
async def preload_sources():
    return {
        "memeSources": await load_file("memeSources.txt"),
    }


# Preload sources using an event loop
loop = asyncio.get_event_loop()
sources = loop.run_until_complete(preload_sources())
memeSources = sources["memeSources"]

# Bot setup with intents and message caching
intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(
    command_prefix=".", intents=intents, case_insensitive=True, max_messages=1000
)

# Remove the default help command
bot.remove_command("help")

# Executor for blocking operations
executor = ThreadPoolExecutor(max_workers=4)


# notes: Logs when the bot is online and ready to accept commands
@bot.event
async def on_ready():
    logger.info("'QUARGLE' initialized.")


# notes: Sets up the HTTP session and executor when the bot starts, ensuring async resources are ready
@bot.event
async def setup_hook():
    bot.http_session = aiohttp.ClientSession()
    bot.executor = executor


# notes: Checks if a user has a specific permission, sends error message if not
async def check_permissions(ctx, permission):
    if not getattr(ctx.author.guild_permissions, permission):
        await ctx.send("You lack permission!", delete_after=2)
        return False
    return True


# notes: Clears a specified number of messages in the channel (capped at 100), requires manage_messages permission
@bot.command()
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount: int):
    if amount > 100:
        await ctx.send("I WON'T DELETE MORE THAN 100 MESSAGES!!!!", delete_after=2)
        return

    await ctx.send(f"Deleting {amount} messages... Please wait.", delete_after=2)

    deleted_count = 0
    while deleted_count < amount:
        try:
            # Delete messages in batches of 10 with a short delay to avoid rate limiting
            batch_size = min(10, amount - deleted_count)
            deleted = await ctx.channel.purge(limit=batch_size)
            deleted_count += len(deleted)

            if deleted:
                await asyncio.sleep(2)  # Small delay between deletions

        except discord.errors.HTTPException as e:
            # If rate-limited, we check the retry-after time and wait accordingly
            if e.code == 429:  # Rate limit error code
                retry_after = int(e.retry_after)  # Time in seconds to wait
                await ctx.send(
                    f"Rate-limited. Retrying in {retry_after} seconds...",
                    delete_after=5,
                )
                await asyncio.sleep(retry_after)  # Wait the suggested time and retry
            else:
                await ctx.send(
                    "An error occurred while deleting messages.", delete_after=2
                )
                return

    await ctx.send(f"Deleted {deleted_count} messages.", delete_after=2)


# Error handling for missing permissions
@clear.error
async def clear_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send(
            "You need Manage Messages permission to use this command!", delete_after=2
        )


@bot.command()
async def debug(ctx):
    await ctx.send("Regular Debug", delete_after=1)


@bot.command()
async def update(ctx):
    message = await ctx.send(
        "Bot is prepping for updates... <a:4704loadingicon:1246520222844977252>",
        delete_after=4,
    )
    await asyncio.sleep(5)  # Ensures the message is deleted before closing
    await bot.close()


# notes: Fetches and displays a user's profile picture as an embed, defaults to command issuer if no member specified
@bot.command()
async def getpfp(ctx, member: Member = None):
    member = member or ctx.author
    embed = Embed(title=str(member), url=member.display_avatar)
    embed.set_image(url=member.display_avatar)
    await ctx.send(embed=embed)


# notes: Retrieves and displays a 3-day weather forecast for a specified city using python_weather
@bot.command()
async def weather(ctx, *, city=""):
    if not city:
        await ctx.send("City is missing", delete_after=1)
        return
    async with python_weather.Client(unit=python_weather.IMPERIAL) as wc:
        weather = await wc.get(city)
        current_temp = weather.temperature
        forecast_msg = [f"The current temperature in {city} is {current_temp}°F."]
        forecast_msg.extend(
            f"{daily.date.strftime('%m/%d')}: High: {daily.highest_temperature}°F, "
            f"Low: {daily.lowest_temperature}°F, Sunset: {daily.sunset.strftime('%I:%M %p')}"
            for daily in weather.daily_forecasts
        )
        await ctx.send("\n".join(forecast_msg))


# notes: Fetches a random meme from a predefined list of Reddit sources, retries up to 3 times if content is invalid
@bot.command()
async def meme(ctx):
    await ctx.message.delete(delay=1)
    embed = Embed()
    max_retries = 3

    for attempt in range(max_retries):
        meme_url = random.choice(memeSources)
        try:
            async with bot.http_session.get(meme_url) as r:
                if r.status == 404:
                    logger.warning(f"404 for {meme_url}")
                    continue
                res = await r.json()
            img_num = random.randint(0, min(24, len(res["data"]["children"]) - 1))
            data = res["data"]["children"][img_num]["data"]

            if (
                data["is_video"]
                or data["domain"] == "Youtube"
                or not data["is_reddit_media_domain"]
            ):
                continue

            embed.title = data["title"]
            embed.set_image(url=data["url"])
            await ctx.send(embed=embed)
            return
        except Exception as e:
            logger.error(f"Meme fetch error: {e}")
            if attempt == max_retries - 1:
                await ctx.send("Failed to fetch meme.", delete_after=1)


# notes: Replies to a referenced message with a GIF from Tenor based on sanitized message content
@bot.command()
async def reaction(ctx):
    await ctx.message.delete(delay=1)

    if ctx.message.reference:
        referenced_message = await ctx.channel.fetch_message(
            ctx.message.reference.message_id
        )
        original_message = referenced_message.content
        username = referenced_message.author.name
        embed = discord.Embed(title="", description="")

        # Define word lists in-code
        disallowed_words = {"cunt", "nazi", "retard"}
        replacements = {"cunt": "jerk", "nazi": "creep", "retard": "goof"}

        # Replace disallowed words
        words = original_message.split()
        sanitized_message = []
        for word in words:
            word_lower = word.lower()
            if word_lower in disallowed_words:
                replacement = replacements.get(word_lower, "jerk")
                if word.isupper():
                    sanitized_message.append(replacement.upper())
                elif word[0].isupper():
                    sanitized_message.append(replacement.capitalize())
                else:
                    sanitized_message.append(replacement)
            else:
                sanitized_message.append(word)
        sanitized_message = " ".join(sanitized_message)

        # Scrape Tenor search page
        search_term = urllib.parse.quote(sanitized_message)
        tenor_url = f"https://tenor.com/search/{search_term}-gifs"

        async with aiohttp.ClientSession() as session:
            async with session.get(tenor_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    gif_img = soup.find("img", src=lambda x: x and ".gif" in x)
                    if gif_img and gif_img["src"]:
                        gif_url = gif_img["src"]
                    else:
                        await ctx.send("No GIFs found on the page.", delete_after=2)
                        return
                else:
                    await ctx.send("Failed to load Tenor search page.", delete_after=2)
                    return

        embed.title = f"{username}: {original_message}"
        embed.set_image(url=gif_url)

        # Replying to the referenced message
        await referenced_message.reply(embed=embed)
        return
    else:
        await ctx.send("No message was referenced.", delete_after=2)


# notes: Uploads all attachments from a message to the OurMemes folder concurrently
@bot.command()
async def upload(ctx):
    if not ctx.message.attachments:
        await ctx.send("No attachments!", delete_after=4)
        return

    tasks = [
        asyncio.create_task(save_attachment(att)) for att in ctx.message.attachments
    ]
    await asyncio.gather(*tasks)
    await ctx.send(f"All {len(tasks)} file(s) uploaded!", delete_after=10)


# notes: Saves a single attachment to the OurMemes folder asynchronously
async def save_attachment(attachment):
    async with bot.http_session.get(attachment.url) as resp:
        if resp.status == 200:
            filename = os.path.join(
                "OurMemes", os.path.basename(urlparse(attachment.url).path)
            )
            async with aiofiles.open(filename, "wb") as f:
                await f.write(await resp.read())


# notes: Sends a random image or video from OurMemes folder with a random title from Oldwordlist.txt
@bot.command()
async def ourmeme(ctx, media_type: str = None):
    valid_exts = {"image": (".png", ".jpg", ".gif"), "video": (".mp4", ".mov", ".mkv")}
    exts = valid_exts.get(
        media_type.lower() if media_type else None,
        valid_exts["image"] + valid_exts["video"],
    )

    files = [f for f in os.listdir("OurMemes") if f.lower().endswith(exts)]
    if not files:
        await ctx.send(f"No {media_type or 'memes'} found!")
        return

    file_path = os.path.join("OurMemes", random.choice(files))
    async with aiofiles.open("Oldwordlist.txt", "r") as f:
        words = [line.strip() async for line in f if line.strip()]
    title = random.choice(words) if words else "Random Meme"

    file = discord.File(file_path)
    if file_path.lower().endswith(valid_exts["image"]):
        embed = Embed(title=title, color=discord.Color.blue())
        embed.set_image(url=f"attachment://{os.path.basename(file_path)}")
        await ctx.send(embed=embed, file=file)
    else:
        await ctx.send(content=title, file=file)
    await ctx.message.delete(delay=1)


# AI Features Setup
BOT_IDENTITY = "I am QUARGLE, your AI-powered assistant! I assist users in this Discord server by answering questions, generating ideas, and helping with tasks. I am friendly, knowledgeable, and always here to help!"
conversation_history = {}
user_preferences = {}


# notes: Sets a custom context for the AI to tailor its responses for a specific user
@bot.command()
async def setcontext(ctx, *, new_context: str):
    user_preferences[ctx.author.id] = new_context
    await ctx.send(f"Context updated: {new_context}")


# notes: Interactive AI chat using OpenAI's GPT-4o, maintains conversation history per user
@bot.command()
async def QUARGLE(ctx, *, inputText: str):
    user_id = ctx.author.id
    openai.api_key = OPENAI_GPT_TOKEN

    if user_id not in conversation_history:
        role = next(
            (r.name for r in ctx.author.roles if r.name != "@everyone"), "Member"
        )
        system_msg = {
            "role": "system",
            "content": f"{BOT_IDENTITY} Assisting a {role}. {user_preferences.get(user_id, '')}",
        }
        conversation_history[user_id] = [system_msg]

    conversation_history[user_id].append({"role": "user", "content": inputText})
    conversation_history[user_id] = conversation_history[user_id][-10:]

    async with bot.executor:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model="gpt-4o", messages=conversation_history[user_id]
            ),
        )

    bot_response = response.choices[0].message.content
    conversation_history[user_id].append({"role": "assistant", "content": bot_response})
    await ctx.send(bot_response)


# notes: Generates an image using DALL-E 3 based on user input and displays it as an embed
@bot.command()
async def imagine(ctx, *, inputText: str):
    openai.api_key = OPENAI_GPT_TOKEN
    loading_msg = await ctx.send(
        "Processing <a:4704loadingicon:1246520222844977252>", delete_after=1
    )

    async with bot.executor:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai.images.generate(
                model="dall-e-3",
                prompt=inputText,
                size="512x512",
                n=1,
                response_format="url",
            ),
        )

    embed = Embed(title=inputText, url=response.data[0].url)
    embed.set_image(url=response.data[0].url)
    await ctx.send(embed=embed)
    logger.info(f"Generated image for '{inputText}': {response.data[0].url}")


# Help Menu Setup
COMMAND_CATEGORIES = {
    "Utilities": {
        "clear": "Clears a specified number of messages (requires manage messages permission)",
        "getpfp": "Displays a user's profile picture",
        "weather": "Shows a 3-day weather forecast for a given city",
    },
    "Memes": {
        "meme": "Fetches a random meme from Reddit",
        "reaction": "Replies to a message with a relevant GIF",
        "ourmeme": "Shares a random image or video from the OurMemes folder",
        "upload": "Uploads attachments to the OurMemes folder",
    },
    "AI Features": {
        "setcontext": "Sets the AI's context for responses",
        "QUARGLE": "Interact with the AI assistant",
        "imagine": "Generates an image using DALL-E 3",
    },
}

COLORS = {
    "Utilities": discord.Color.blue(),
    "Memes": discord.Color.green(),
    "AI Features": discord.Color.purple(),
}


# notes: Provides a visually appealing, paginated help menu with reaction navigation, overrides default help
@bot.command(name="help")
async def help_command(ctx):
    pages = []

    for category, commands_dict in COMMAND_CATEGORIES.items():
        embed = Embed(
            title=f"QUARGLE-HELP - {category}",
            color=COLORS.get(category, discord.Color.blue()),
            description=f"Commands for {category.lower()}.",
        )
        embed.set_thumbnail(url=bot.user.avatar.url if bot.user.avatar else None)
        embed.set_footer(
            text=f"Page {len(pages) + 1}/{len(COMMAND_CATEGORIES)} | Prefix: ."
        )

        for cmd, desc in commands_dict.items():
            embed.add_field(name=f".{cmd}", value=desc, inline=False)
        pages.append(embed)

    current_page = 0
    message = await ctx.send(embed=pages[current_page])

    await message.add_reaction("⬅️")
    await message.add_reaction("➡️")
    await message.add_reaction("❌")

    def check(reaction, user):
        return (
            user == ctx.author
            and str(reaction.emoji) in ["⬅️", "➡️", "❌"]
            and reaction.message.id == message.id
        )

    while True:
        try:
            reaction, user = await bot.wait_for(
                "reaction_add", timeout=60.0, check=check
            )

            if str(reaction.emoji) == "➡️" and current_page < len(pages) - 1:
                current_page += 1
                await message.edit(embed=pages[current_page])
            elif str(reaction.emoji) == "⬅️" and current_page > 0:
                current_page -= 1
                await message.edit(embed=pages[current_page])
            elif str(reaction.emoji) == "❌":
                await message.delete()
                break

            await message.remove_reaction(reaction, user)

        except asyncio.TimeoutError:
            await message.clear_reactions()
            break


# notes: Cleans up resources (HTTP session and executor) when the bot shuts down
async def close():
    await bot.http_session.close()
    bot.executor.shutdown()


bot.on_close = close

bot.run(BOT_TOKEN)
