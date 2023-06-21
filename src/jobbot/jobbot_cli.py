import argparse
import os
import sys
import logging
from typing import List

from jobbot.scraper import JobBotScraper
from jobbot.spiders import MinnesotaWorksSpider

__version__ = "0.1.0"

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class JobBotCli:
    """Scrape sources for job postings and candidates"""

    def __init__(self):
        self._arg_parser = argparse.ArgumentParser(description=JobBotCli.__doc__)
        self._arg_parser.set_defaults(func=self.version)

        cmd_parser = self._arg_parser.add_subparsers()

        scraper_parser = cmd_parser.add_parser(
            "scrape",
            help="Run scraper",
        )
        scraper_parser.set_defaults(func=self.run_scraper)
        scraper_parser.add_argument(
            "-t",
            dest="tags",
            type=str,
            nargs="+",
            help="List of tags to run",
        )
        scraper_parser.add_argument(
            "-f",
            dest="catalog_file",
            type=str,
            default="catalog.yml",
            help="The path to the yml configuration file",
        )

        version = cmd_parser.add_parser(
            "version",
            help="Print the version of the CLI",
        )
        version.set_defaults(func=self.version)

        view = cmd_parser.add_parser(
            "view",
            help="View results",
        )
        view.set_defaults(func=self.version)

    @staticmethod
    def run_scraper(args):
        """Run the scraper with the given arguments"""
        _logger.info("Running scraper")
        JobBotScraper(
            catalog_file=args.catalog_file,
            tags=args.tags,
            spiders=[MinnesotaWorksSpider],
        ).run()

    def execute(self, argv: List[str]) -> int:
        """Execute the CLI with the given arguments"""
        args = self._arg_parser.parse_args(argv)
        func = args.func
        del args.func
        return func(args)

    @staticmethod
    def version(*_args):
        """Current CLI version"""
        _logger.info("jobbot-%s", __version__)

    @classmethod
    def main(cls) -> int:
        """Pull values from the environment and run the CLI"""
        orig_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + orig_path
        return cls().execute(sys.argv[1:])


if __name__ == "__main__":
    JobBotCli.main()
