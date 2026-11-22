import argparse
import logging
import time

from src.data import HelocDataset
from src.eda import ExploratoryAnalysis
from src.evaluation import ModelEvaluator
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
    logging.getLogger("interpret").setLevel(logging.WARNING)
    start = time.perf_counter()

    logger.info("Step 1/5: loading data")
    dataset = HelocDataset.from_openml()

    logger.info("Step 2/5: splitting data")
    split = dataset.split()

    logger.info("Step 3/5: exploratory analysis")
    ExploratoryAnalysis(split.X_train, split.y_train).run()

    logger.info("Step 4/5: training models")
    trainer = ModelTrainer(split, dataset.special_value_indicators(), retrain=args.retrain)
    models = trainer.train_all()

    logger.info("Step 5/5: evaluating models on the test set")
    ModelEvaluator(models, split).run()

    logger.info("Done in %.1f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
