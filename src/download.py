import datetime
import time
from tqdm import tqdm
from telethon.sync import TelegramClient
from collections import defaultdict
from datetime import timedelta


class Downloader:
    def __init__(
        self,
        api_id,
        api_hash,
        phone,
        channels=[
            'rian_ru',
            'bbbreaking',
            'ENews112',
            'kommersant',
            'rt_russian',
            'breakingmash',
            'moscowmap',
            'moscowachplus',
            'rbc_news',
            'izvestia',
            'tass_agency',
            'vestiru24',
            'gazetaru',
            'interfaxonline'
        ],
        time2wait=1
    ):
        self.client = TelegramClient(phone, api_id, api_hash)
        self.client.start()
        self.channels = channels
        self.api_id = api_id
        self.phone = phone
        self.api_hash = api_hash
        self.time2wait = time2wait

    async def download_messages_by_last_n_days(self, n=2):
        channel2messages_fin = defaultdict(list)
        today = datetime.datetime.today()
        for i in tqdm(range(n)):
            new_date = datetime.datetime.strftime(
                today - timedelta(days=i), '%Y-%m-%d')
            channel2messages = await self.download_messages_by_date(new_date)
            for channel, messages in channel2messages.items():
                channel2messages_fin[channel].extend(channel2messages[channel])

        return channel2messages_fin

    async def download_messages_by_date(self, date):
        channel2messages = defaultdict(list)
        start_date_string = date
        start_date = datetime.datetime.strptime(start_date_string, '%Y-%m-%d')
        tommorow_start_date = start_date + timedelta(days=1)
        async with TelegramClient('name', self.api_id, self.api_hash) as client:
            for channel in self.channels:
                entity = await client.get_entity(channel)
                id_channel = entity.id

                async for message in client.iter_messages(
                    id_channel,
                    offset_date=tommorow_start_date
                ):
                    # print(message.date, message.id)
                    if start_date_string in str(message.date):
                        channel2messages[channel].append(message.to_dict())
                    else:
                        time.sleep(self.time2wait)
                        break
        return channel2messages

    async def update_messages(self, channel2max_message_id):
        channel2messages = defaultdict(list)
        async with TelegramClient('name', self.api_id, self.api_hash) as client:
            for channel, max_id in channel2max_message_id.items():
                entity = await client.get_entity(channel)
                count = 0
                id_channel = entity.id
                async for message in client.iter_messages(
                    id_channel,
                    min_id=channel2max_message_id[channel]
                ):
                    # print(message.date, message.id)
                    count += 1
                    channel2messages[channel].append(message.to_dict())
                print(f"Channel {channel} has new {count} messages!")
                time.sleep(self.time2wait)

        return channel2messages
