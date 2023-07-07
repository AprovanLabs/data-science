from jobbot.scraper.jobbot_spider import JobBotSpider
from jobbot.scraper.items import CandidateItem, JobHistoryItem
import datetime
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import uuid
import logging

from itemloaders.processors import MapCompose, TakeFirst
from scrapy.utils.response import open_in_browser
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule
from scrapy.loader import ItemLoader
from scrapy.http import FormRequest, Request

_MP_UNIQUE_ID = "ctl00"
_FORM_ITEM_PREFIX = f"{_MP_UNIQUE_ID}$SubContentPlaceholder$"

_SELECTED_VALUE = "on"
_HEADERS = {
    "dnt": "1",
    "accept-encoding": "gzip, deflate, sdch, br",
    "accept-language": "en-US,en;q=0.8",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "cache-control": "max-age=0",
    "Sec-Fetch-User": "?1",
    "Origin": "https://minnesotaworks.net",
    "Referer": "https://minnesotaworks.net/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "sec-ch-ua-platform": "macOS",
}


class JobOrder(Enum):
    ANY = 0
    BUSINESS_DEVELOPMENT_ASSOCIATE = "A13149811"  # Business Development Associate
    ELECTRONICS_TECHNITIAN = "B13262712"  # Electronics Technician
    ESTIMATOR = "X13428484"  # Estimator
    JUNIOR_ACCOUNTANT = "U13428481"  # Junior Accountant
    MACHINE_OPERATOR1 = "N13142307"  # Machine Operator
    MACHINE_OPERATOR2 = "U13142314"  # Machine Operator
    PICKER_PACKER = "G13447154"  # Picker/Packer
    SALES_EXECUTIVE1 = "N13453920"  # Sales Executive
    SALES_EXECUTIVE2 = "R13453948"  # Sales Executive
    SOLDERING_TECHNITIAN = "A13142620"  # Soldering Technician
    TECHNICAL_RECRUITER = "Z13149810"  # Technical Recruiter
    TRUSS_ASSEMBLER = "E13447153"  # Truss Assembler


class JobLevel(Enum):
    LESS_THAN_HIGH_SCHOOL = 0
    HIGH_SCHOOL = 1
    SOME_COLLEGE = 2
    VOCATIONAL_SCHOOL = 3
    ASSOCIATES_DEGREE = 4
    BACHELORS_DEGREE = 5
    MASTERS_DEGREE = 6
    DOCTORAL_DEGREE = 7


class WorkWeek(Enum):
    ANY = 3
    FULL_TIME = 1
    PART_TIME = 2


class WorkShift(Enum):
    ANY = 0
    FIRST = 1  # Day
    SECOND = 2  # Evening
    THIRD = 3  # Night
    SPLIT = 4  # Split
    ROTATING = 5  # Rotating
    OTHER = 6  # Other


class WorkType(Enum):
    REGULAR = 1
    REGULAR_OR_TEMPORARY = 3
    TEMPORARY = 2
    ANY = 0


@dataclass()
class MinnesotaWorksOptions:
    job_levels: Optional[List[JobLevel]]
    search_term: str
    zip_code: str
    work_week: WorkWeek
    work_shift: WorkShift
    work_type: WorkType
    job_order: JobOrder = 0
    military_search: bool = False


class MinnesotaWorksSpider(JobBotSpider):
    """Scrape Minnesota Works for job postings"""

    id = "minnesota_works"
    title = "Minnesota Works Jobs"
    start_urls = ["https://www.minnesotaworks.net/"]
    tags = ["location:minnesota", "category:job"]
    options: MinnesotaWorksOptions

    allowed_domains = ["minnesotaworks.net"]
    download_delay = 1.5

    # Rules for horizontal and vertical crawling
    rules = (
        Rule(LinkExtractor(restrict_xpaths='//*[contains(@class,"next")]')),
        Rule(
            LinkExtractor(restrict_xpaths='//*[@itemprop="url"]'), callback="parse_item"
        ),
    )

    def _get_form_value(self, name, response):
        return response.css(f"input#{name}::attr(value)").extract_first()

    def _get_page_number(self, response):
        page_number = response.xpath(
            self._get_form_value("ctl00_SubContentPlaceholder_ResumeIndex", response)
        ).extract_first()
        return int(float(page_number))

    def _id_xpath_value(self, id: str):
        return f'.//*[@id="{id}"]/text()'

    # Start with a login request
    def start_requests(self):
        """Navigate to the login page"""
        yield Request(
            "https://www.minnesotaworks.net/",
            headers=_HEADERS,
            callback=self.login,
        )

    def login(self, response):
        """Login with username/password"""
        username, password = self.get_credentials(["username", "password"])
        viewstate = self._get_form_value("__VIEWSTATE", response)
        eventvalidation = self._get_form_value("__EVENTVALIDATION", response)
        previouspage = self._get_form_value("__PREVIOUSPAGE", response)

        formdata = {
            "LoginVW$MainLog$UserName": username,
            "LoginVW$MainLog$Password": password,
            "__VIEWSTATE": viewstate,
            "__PREVIOUSPAGE": previouspage,
            "__EVENTVALIDATION": eventvalidation,
            "__EVENTTARGET": "",
            "__EVENTARGUMENT": "",
            "__VIEWSTATEGENERATOR": "CA0B0334",
            "cookieexists": "false",
            "LoginVW$MainLog$LoginImageButton.x": "24",
            "LoginVW$MainLog$LoginImageButton.y": "5",
        }

        yield FormRequest.from_response(
            response,
            formname="MasterForm",
            formdata=formdata,
            callback=self.parse,
        )

    def parse(self, _response):
        logging.info("Starting search")
        zip_code = self.options.get("zip_codes").pop()

        if zip_code is None:
            return

        logging.info("Searching zip code: %s", zip_code)
        yield Request(
            "https://www.minnesotaworks.net/Search.aspx?Searchtype=resume",
            callback=self.search_postings,
            cb_kwargs={"zip_code": zip_code},
        )

    def search_postings(self, response, zip_code):
        yield FormRequest.from_response(
            response,
            formname="aspnetForm",
            formdata={
                k: str(v)
                for k, v in {
                    f"{_FORM_ITEM_PREFIX}JobOrderSearchDdl": self.options.get(
                        "job_order"
                    ),
                    f"{_FORM_ITEM_PREFIX}ZipcodeTextBox": zip_code,
                    f"{_FORM_ITEM_PREFIX}Keywordtext": self.options.get(
                        "search_term", ""
                    ),
                    f"{_FORM_ITEM_PREFIX}WorkweekDdl": self.options.get("work_week"),
                    f"{_FORM_ITEM_PREFIX}WorkShiftDdl": self.options.get("work_shift"),
                    f"{_FORM_ITEM_PREFIX}WorkTypeDdl": self.options.get("work_type"),
                    f"{_FORM_ITEM_PREFIX}Location": "LocationButton",
                    f"{_FORM_ITEM_PREFIX}SearchButton": "Search",
                    **{
                        f"{_FORM_ITEM_PREFIX}EducationLst${job_level}": _SELECTED_VALUE
                        for job_level in self.options.get("job_levels") or []
                    },
                    "__VIEWSTATE": self._get_form_value("__VIEWSTATE", response),
                }.items()
            },
            callback=self.look_at_first_search_result,
        )

    def look_at_first_search_result(self, response):
        eventvalidation = self._get_form_value("__EVENTVALIDATION", response)
        viewstate = self._get_form_value("__VIEWSTATE", response)

        yield FormRequest(
            "https://www.minnesotaworks.net/SearchResult.aspx",
            formdata={
                "__VIEWSTATE": viewstate,
                "__EVENTVALIDATION": eventvalidation,
                "__VIEWSTATEENCRYPTED": "",
                "__LASTFOCUS": "",
                "__VIEWSTATEGENERATOR": "F8B5C114",
                "__EVENTARGUMENT": "",
                "__EVENTTARGET": f"{_FORM_ITEM_PREFIX}SearchResultsGrid$ctl03$ResumeIDButton",
                f"{_FORM_ITEM_PREFIX}GridResultsManagerControl2$GridResultsManagerControl2PageSize": "50",
                f"{_FORM_ITEM_PREFIX}ABclickHdn": "",
            },
            callback=self.parse_posting,
        )

    def parse_posting(self, response):
        selector = response.xpath('//div[@id="content"]')
        yield from self.parse_item(selector, response)

        page_number = self._get_page_number(response)
        logging.info("Page #%s", page_number)

        is_last_page = response.css(
            "input#ctl00_SubContentPlaceholder_NextButton1::attr(disabled)"
        ).extract_first()
        logging.info(is_last_page)
        is_last_page = is_last_page == "disabled"

        if is_last_page:
            yield from self.parse(response)
            return

        yield FormRequest.from_response(
            response,
            formdata={
                f"{_FORM_ITEM_PREFIX}ResumeIndex": str(page_number),
                f"{_FORM_ITEM_PREFIX}NextButton1": "Next Resume >",
                "__VIEWSTATE": self._get_form_value("__VIEWSTATE", response),
            },
            callback=self.parse_posting,
        )

    def parse_item(self, selector, response):
        item_loader = ItemLoader(item=CandidateItem(), selector=selector)

        id = str(uuid.uuid4())

        item_loader.add_value("id", id)
        resume_id = self._id_xpath_value(
            "ctl00_SubContentPlaceholder_NoContactResumeLbl"
        )
        item_loader.add_xpath(
            "resume_id",
            resume_id,
            MapCompose(lambda x: x.split(": ")[-1]),
        )
        item_loader.add_xpath(
            "name", self._id_xpath_value("ctl00_SubContentPlaceholder_NameLbl")
        )
        item_loader.add_xpath(
            "street", self._id_xpath_value("ctl00_SubContentPlaceholder_Addr1Lbl")
        )
        item_loader.add_xpath(
            "city_state_zip_code",
            self._id_xpath_value("ctl00_SubContentPlaceholder_CityStateZipLbl"),
        )
        item_loader.add_xpath(
            "phone_number", self._id_xpath_value("ctl00_SubContentPlaceholder_PhoneLbl")
        )
        item_loader.add_xpath(
            "email", self._id_xpath_value("ctl00_SubContentPlaceholder_EmailLbl")
        )

        for job_title, job_description in zip(
            selector.xpath(".//h5//text()"),
            selector.xpath(".//h5//following-sibling::*"),
        ):
            (
                company,
                location,
                dates,
                description,
                *_,
            ) = job_description.xpath(
                # Ignores <br /> elements
                ".//text()"
            ) + ([None] * 4)

            try:
                company = company.get()
            except Exception:
                pass
            try:
                location = location.get()
            except Exception:
                pass
            try:
                description = description.get()
            except Exception:
                pass

            # e.g. 'Start Date: January 2015 End Date: May 2015'
            start_date, end_date = None, None
            try:
                start_date, end_date = dates.get().split("End Date:")
                end_date = end_date.strip()
                start_date = start_date.split("Start Date:")[-1].strip()
            except Exception:
                pass

            jh_item_loader = ItemLoader(item=JobHistoryItem())

            jh_item_loader.add_value("id", str(uuid.uuid4()))
            jh_item_loader.add_value("candidate_id", id)
            jh_item_loader.add_value("spider", self.name)
            jh_item_loader.add_value("timestamp", datetime.datetime.now())
            jh_item_loader.add_value("title", job_title.get())
            jh_item_loader.add_value("company", company)
            jh_item_loader.add_value("location", location)
            jh_item_loader.add_value("start_date", start_date)
            jh_item_loader.add_value("end_date", end_date)
            jh_item_loader.add_value("description", description)

            yield jh_item_loader.load_item()

        item_loader.add_xpath(
            "notes", self._id_xpath_value("ctl00_SubContentPlaceholder_NotesLbl")
        )

        item_loader.add_value("url", response.url)
        item_loader.add_value("spider", self.name)
        item_loader.add_value("timestamp", datetime.datetime.now())

        yield item_loader.load_item()
