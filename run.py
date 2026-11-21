import argparse
import logging
import time

from src.data import HelocDataset
from src.eda import ExploratoryAnalysis
from src.models import ModelTrainer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HELOC interpretability experiment.")
    parser.add_argument(
        "--retrain", action="store_true", help="retrain all models even if saved ones exist"
    )
    args = parser.parse_args()

    # force=True: importing PiML already attaches a handler to the root logger.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    start = time.perf_counter()

    logger.info("Step 1/4: loading data")
    dataset = HelocDataset.from_openml()

    logger.info("Step 2/4: splitting data")
    split = dataset.split()

    logger.info("Step 3/4: exploratory analysis")
    ExploratoryAnalysis(split.X_train, split.y_train).run()

    logger.info("Step 4/4: training models")
    ModelTrainer(split, dataset.special_value_indicators(), retrain=args.retrain).train_all()

    logger.info("Done in %.1f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
