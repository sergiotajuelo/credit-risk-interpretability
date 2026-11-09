import logging
import time

from src.data import HelocDataset
from src.eda import ExploratoryAnalysis

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    start = time.perf_counter()

    logger.info("Step 1/3: loading data")
    dataset = HelocDataset.from_openml()

    logger.info("Step 2/3: splitting data")
    split = dataset.split()

    logger.info("Step 3/3: exploratory analysis")
    ExploratoryAnalysis(split.X_train, split.y_train).run()

    logger.info("Done in %.1f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
