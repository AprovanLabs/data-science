from scrapy.item import Item, Field
from itemloaders.processors import TakeFirst


class JobHistoryItem(Item):
    id = Field(output_processor=TakeFirst())
    candidate_id = Field(output_processor=TakeFirst())
    spider = Field(output_processor=TakeFirst())
    timestamp = Field(output_processor=TakeFirst())
    title = Field(output_processor=TakeFirst())
    company = Field(output_processor=TakeFirst())
    location = Field(output_processor=TakeFirst())
    start_date = Field(output_processor=TakeFirst())
    end_date = Field(output_processor=TakeFirst())
    description = Field(output_processor=TakeFirst())


class CandidateItem(Item):
    id = Field(output_processor=TakeFirst())
    spider = Field(output_processor=TakeFirst())
    timestamp = Field(output_processor=TakeFirst())
    resume_id = Field(output_processor=TakeFirst())
    url = Field(output_processor=TakeFirst())
    name = Field(output_processor=TakeFirst())
    street = Field(output_processor=TakeFirst())
    city_state_zip_code = Field(output_processor=TakeFirst())
    phone_number = Field(output_processor=TakeFirst())
    email = Field(output_processor=TakeFirst())
    notes = Field(output_processor=TakeFirst())
