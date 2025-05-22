import pickle
import datetime
import os

from telethon.sync import TelegramClient
from collections import defaultdict
from tqdm import tqdm


# TODO: 1. Сделать сохранение по датам.
# TODO: 2. Чтобы найти нужный день берем data_offset
# TODO: 3. Сделать кеш.

if __name__ == '__main__':
    api_id = os.environ['APIID']
    api_hash = os.environ['APIHASH']
    phone = os.environ['PHONE']

    client = TelegramClient(phone, api_id, api_hash)
    client.start()

    channel2messages = defaultdict(list)
    limit = 4000
    channels = ['@mash', '@rucriminalinfo', 'bbbreaking', 'ENews112']
    with TelegramClient('name', api_id, api_hash) as client:
        for channel in tqdm(channels):
            entity = client.get_entity(channel)
            id_channel = entity.id

            messages = client.get_messages(id_channel, limit=limit)
            i = 0
            while i < limit:
                channel2messages[channel].append(
                    (messages[i].text, messages[i].date))
                if (datetime.datetime.now(tz=datetime.timezone.utc) - messages[i].date).days > 30:
                    print("I am in break")
                    break
                i += 1

    with open(f"data/{str(datetime.datetime.now())}[:10]_channel2messages.pickle", "wb") as file:
        pickle.dump(channel2messages, file)
