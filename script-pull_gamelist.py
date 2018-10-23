import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import BGG_csvsetting

# Build python subclass from the scrapy class Spider which will crawl through URls
class BGGSpider(scrapy.Spider):
    name = "bgg_spider"
    home_url = 'https://www.boardgamegeek.com'
    start_page = str(1)
    start_urls = [home_url+'/browse/boardgame/page/'+start_page] #List of games on BGG, first page of 100
    print("Starting at: " + start_page)

    custom_settings = {
        'DOWNLOAD_DELAY': 2,  # 2 second delay
#        'ITEM_PIPELINES': {'__main__.CSVExportPipeline': 1},
        'FEED_EXPORTERS' : {'csv' : 'BGG_csvsetting.BGGCsvItemExporter'},
        'FEED_FORMAT' : 'csv',
        'CSV_DELIMITER' : ';',
        'FEED_URI' : 'bgg_id_output.csv',
#        'FEED_EXPORT_FIELDS': ['Game','GameID']}
        'FEED_EXPORT_FIELDS': ['Game', 'BGG Rank','GameID']}

    def parse(self, response):
        # Define the selector (pattern) to find the relevant element on the page
        # BGG starting URL shows a table of 100 games; the html elements are in the next cell
        gamelist_ID = response.xpath('//tr[@id = "row_"]//td[contains(@class,"collection_objectname")]//a[contains(@href, "/boardgame/")]/@href').extract()
        gamelist_rank = response.xpath('//tr[@id = "row_"]//td[contains(@class,"collection_rank")]//a/@name').extract()
        gamelist_name = response.xpath('//tr[@id = "row_"]//td[contains(@class,"collection_objectname")]//a[contains(@href, "/boardgame/")]/text()').extract()

        #print(gamelist_name)
        #print(gamelist_ID)
        allIDs = {}

#        for game in zip(gamelist_name, gamelist_ID):
        if len(gamelist_rank) > 0:
            for game in zip(gamelist_name[0:len(gamelist_rank)], gamelist_rank, gamelist_ID[0:len(gamelist_rank)]):
                yield {
                    'Game':game[0],
                    'BGG Rank':game[1],
                    'GameID':game[2].split("/")[2]
                }

            # Go to next page
            next_page = response.xpath('//a[contains(@title, "next page")]/@href').extract_first()
            if next_page:
                home_url = 'https://www.boardgamegeek.com'
                print(["Going to: "+ home_url + next_page])
                yield scrapy.Request(
                    response.urljoin([home_url+next_page][0]),
                    callback = self.parse
            )


s = get_project_settings()
 # Starts a Twisted reactor, configuring logging and setting shutdown handlers
process = CrawlerProcess(s)
process.crawl(BGGSpider) #Load the spider
process.start()
