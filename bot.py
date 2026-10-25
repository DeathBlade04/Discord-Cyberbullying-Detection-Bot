import os
import csv
import traceback
import discord
from datetime import datetime, timezone
from dotenv import load_dotenv

from cb_model import CBModel, MeanEmbeddingVectorizer  # noqa: F401 - required for unpickling mevectorizer.pickle when bot.py is __main__
from phishing_model import PhishingLinkModel
from sqli_model import SQLIModel

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

cb_model = CBModel()
pl_model = PhishingLinkModel()
sqli_model = SQLIModel()

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

LOG_PATH = 'moderation_log.csv'


def log_flagged_message(message, reason):
    '''
    Writes a persistent audit record of every flagged message BEFORE it is
    deleted. This exists specifically because flagged messages are removed
    from the channel, so without a log there is no way to later verify what
    content actually triggered a given warning - a live Discord screenshot
    alone can be ambiguous about which message a warning refers to.
    '''
    file_exists = os.path.isfile(LOG_PATH)
    try:
        with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp_utc', 'author', 'channel', 'original_content', 'reason'])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                str(message.author),
                str(message.channel),
                message.content,
                reason,
            ])
    except Exception:
        traceback.print_exc()


async def handle_bad_behaviour(message, reason):
    '''
    logs the flagged message, deletes it, and sends a message to the
    channel mentioning the author
    '''
    author = message.author

    log_flagged_message(message, reason)

    try:
        await message.delete()
    except discord.Forbidden:
        print(f"Missing 'Manage Messages' permission - could not delete message from {author}")
    except discord.NotFound:
        pass  # message already deleted
    except Exception:
        traceback.print_exc()

    try:
        await message.channel.send(f'{author.mention} {reason}')
    except Exception:
        traceback.print_exc()


async def check_cb(content):
    '''
    check for hateful / cyberbullying content in message content
    '''
    try:
        if cb_model.is_cyber_bullying(content):
            return True, 'Hateful/cyberbullying content detected!!'
    except Exception:
        traceback.print_exc()
    return False, ''


async def check_pl(content):
    '''
    check for phishing links in message content. A link is flagged either by
    the trained model (the warning shows the model's confidence) or by the
    brand-impersonation rule (plain warning: the rule's reasoning is not
    shown in the channel, so senders cannot tune links to slip past it).
    '''
    try:
        is_phish, confidence, brand = pl_model.check_text(content)
        if is_phish:
            if brand:
                return True, 'Phishing link detected!!'
            return True, f'Phishing link detected!! (confidence: {confidence:.0%})'
    except Exception:
        traceback.print_exc()
    return False, ''


async def check_sqli(content):
    '''
    check for sql injection in message content
    '''
    try:
        if sqli_model.is_sqli_str(content):
            return True, 'SQL injection detected!!'
    except Exception:
        traceback.print_exc()
    return False, ''


async def do_text_checks(message, content):
    '''
    run all detection checks on text content, in priority order.
    SQLi always runs; the hate-speech check is skipped for messages
    containing a URL, since link text isn't meaningful input for that model.
    '''
    has_url = "http://" in content or "https://" in content or "www." in content

    is_bad, reason = await check_pl(content)
    if is_bad:
        await handle_bad_behaviour(message, reason)
        return

    is_bad, reason = await check_sqli(content)
    if is_bad:
        await handle_bad_behaviour(message, reason)
        return

    if not has_url:
        is_bad, reason = await check_cb(content)
        if is_bad:
            await handle_bad_behaviour(message, reason)
            return


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    n_terms = len(getattr(cb_model, 'hostile_terms', []))
    print('Hostile vocab : ' + (f'ACTIVE ({n_terms} terms)' if n_terms else 'NOT LOADED - cb_model.py is an older version'))
    print(f'Audit log     : {os.path.abspath(LOG_PATH)}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content:
        await do_text_checks(message, message.content)


if __name__ == '__main__':
    if not TOKEN:
        raise RuntimeError(
            "DISCORD_TOKEN not found. Create a .env file next to bot.py "
            "containing: DISCORD_TOKEN=your_token_here"
        )
    client.run(TOKEN)
