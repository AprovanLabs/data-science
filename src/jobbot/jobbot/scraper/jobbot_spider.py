from typing import Any, List, Optional, Mapping

from scrapy.spiders import CrawlSpider


class JobBotSpider(CrawlSpider):
    id: str
    title: str
    credentials: Mapping[str, str] = {}
    options: Mapping[str, Any] = {}
    tags: List[str] = []

    def __init__(
        self,
        *args,
        options: Optional[Mapping[str, Any]] = None,
        credentials: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        if credentials:
            self.credentials = credentials
        if options:
            self.options = options

        super().__init__(*args, name=self.id, **kwargs)

    def get_credentials(self, keys: List[str]) -> List[str]:
        """Get credentials from environment variables"""
        # return [os.environ.get(key) for key in keys]
        return [self.credentials.get(key) for key in keys]
