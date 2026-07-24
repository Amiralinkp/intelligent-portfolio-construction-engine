from abc import ABC


class MarketDataProvider(ABC):

    def get_data(
        self,
        symbols : list[str],
        start_date : str,
        end_date : str):
        
        raise NotImplementedError