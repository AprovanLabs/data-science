from scrapy.item import Item, Field
from typing import List


class JobHistoryItem(Item):
    id = Field()
    candidate_id = Field()
    spider = Field()
    timestamp = Field()
    title = Field()
    company = Field()
    location = Field()
    start_date = Field()
    end_date = Field()
    description = Field()


class CandidateItem(Item):
    id = Field()
    spider = Field()
    timestamp = Field()
    resume_id = Field()
    url = Field()
    name = Field()
    street = Field()
    city_state_zip_code = Field()
    phone_number = Field()
    email = Field()
    notes = Field()
