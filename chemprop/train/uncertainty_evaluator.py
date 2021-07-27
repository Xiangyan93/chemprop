from argparse import Namespace
from scipy import stats
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from typing import Any, Callable, Dict, List


class EvaluationMethod:
    """
    An EvaluationMethod takes in the uncertainty estimates and
    true errors for a set of predictions. From these it is capable
    of producing a numeric evaluation and a visualization.
    """
    def __init__(self):
        self.name = None

    def evaluate(self, data: Dict[str, List[Dict[str, float]]]) -> Any:
        """
        Evaluates the provided uncertainty estimates.

        :param data: A dict storing lists of sorted predictions and errors.
        :return: Some representation of the evaluation.
        """
        pass

    def _visualize(self, task: str, evaluation: Any):
        """
        A helper method which visualizes data given some evaluation.

        :param task: The name of the task being visualized.
        :param evaluation: Some representation of the evaluation.
        """
        pass

    def visualize(self, task: str, data: Dict[str, List[Dict[str, float]]]):
        """
        A wrapper method which evaluates provided data before visualizing it.

        :param task: The name of the task being visualized.
        :param data: A dict storing lists of sorted predictions and errors.
        """
        evaluation = self.evaluate(data)

        sns.set()

        self._visualize(task, evaluation)


class Spearman(EvaluationMethod):
    """
    Computes Spearman's correlation coefficient between the provided
    uncertainty estimates and true errors.
    """
    def __init__(self):
        self.name = 'spearman'

    def evaluate(self, data: Dict[str, List[Dict[str, float]]]) \
            -> Dict[str, float]:
        uncertainty = [set_['uncertainty']
                       for set_ in data['sets_by_uncertainty']]
        error = [set_['error']
                 for set_ in data['sets_by_uncertainty']]

        rho, p = stats.spearmanr(uncertainty, np.abs(error))

        return {'rho': rho, 'p': p}

    def _visualize(self, task: str, evaluation: Dict[str, float]):
        print(task, '-', 'Spearman Rho:', evaluation['rho'])
        print(task, '-', 'Spearman p-value:', evaluation['p'])


class LogLikelihood(EvaluationMethod):
    """
    Computes the log likelihood, average log likelihood, optimal log
    likelihood, and average optimal log likelihood, to produce the
    observed true errors given the provided uncertainty estimates.
    """
    def __init__(self):
        self.name = 'log_likelihood'

    def evaluate(self, data: Dict[str, List[Dict[str, float]]]) \
            -> Dict[str, float]:
        log_likelihood = 0
        optimal_log_likelihood = 0
        for set_ in data['sets_by_uncertainty']:
            # Encourage small standard deviations.
            log_likelihood -= np.log(2 * np.pi * max(0.00001, set_[
                'uncertainty']**2)) / 2
            optimal_log_likelihood -= np.log(2 * np.pi * set_['error']**2) / 2

            # Penalize for large error.
            log_likelihood -= set_['error']**2/(2 * max(0.00001, set_[
                'uncertainty']**2))
            optimal_log_likelihood -= 1 / 2

        return {'log_likelihood': log_likelihood,
                'optimal_log_likelihood': optimal_log_likelihood,
                'average_log_likelihood': log_likelihood / len(data[
                    'sets_by_uncertainty']),
                'average_optimal_log_likelihood': optimal_log_likelihood / len(
                    data['sets_by_uncertainty'])}

    def _visualize(self, task: str, evaluation: Dict[str, float]):
        print(task,
              '-',
              'Sum of Log Likelihoods:',
              evaluation['log_likelihood'])


class MiscalibrationArea(EvaluationMethod):
    """
    Computes the miscalibration area given the provided uncertainty estimates
    and true errors.

    The miscalibration area compares the observed fraction of errors falling
    within 'z' standard deviations of the mean to what is expected for a
    Gaussian random variable with variance equal to the uncertainty prediction.

    The miscalibration area is computed as the area between the true curve of
    observed versus expected fractions and the parity line.
    """
    def __init__(self):
        self.name = 'calibration_auc'

    def evaluate(self, data: Dict[str, List[Dict[str, float]]]) -> Any:
        standard_devs = [np.abs(set_['error'])/set_[
            'uncertainty'] for set_ in data['sets_by_uncertainty']]
        probabilities = [2 * (stats.norm.cdf(
            standard_dev) - 0.5) for standard_dev in standard_devs]
        sorted_probabilities = sorted(probabilities)

        fraction_under_thresholds = []
        threshold = 0

        for i in range(len(sorted_probabilities)):
            while sorted_probabilities[i] > threshold:
                fraction_under_thresholds.append(i/len(sorted_probabilities))
                threshold += 0.001

        # Condition used 1.0001 to catch floating point errors.
        while threshold < 1.0001:
            fraction_under_thresholds.append(1)
            threshold += 0.001

        thresholds = np.linspace(0, 1, num=1001)
        miscalibration = [np.abs(
            fraction_under_thresholds[i] - thresholds[i]) for i in range(
                len(thresholds))]
        miscalibration_area = 0
        for i in range(1, 1001):
            miscalibration_area += np.average([miscalibration[i-1],
                                               miscalibration[i]]) * 0.001

        return {'fraction_under_thresholds': fraction_under_thresholds,
                'thresholds': thresholds,
                'miscalibration_area': miscalibration_area}

    def _visualize(self, task: str, evaluation: Any):
        print(task,
              '-',
              'Miscalibration Area',
              evaluation['miscalibration_area'])

        # Ideal curve.
        plt.plot(evaluation['thresholds'], evaluation['thresholds'])

        # True curve.
        plt.plot(evaluation['thresholds'],
                 evaluation['fraction_under_thresholds'])

        plt.title(task)

        plt.show()


class UncertaintyEvaluator:
    """
    An UncertaintyEvaluator stores the values produced by a
    UQ method experiment and uses some number of EvaluationMethods
    to analyze the results.
    """
    methods = [Spearman(),
               LogLikelihood(),
               MiscalibrationArea()]

    @staticmethod
    def save(val_predictions: np.ndarray,
             val_targets: np.ndarray,
             val_uncertainty: np.ndarray,
             test_predictions: np.ndarray,
             test_targets: np.ndarray,
             test_uncertainty: np.ndarray,
             args: Namespace):
        """
        Sorts predictions made by an uncertainty estimator and saves the
        resulting logs to a file specified in args.

        :param val_predictions: The predicted labels for the validation set.
        :param val_targets: The targets (true labels) for the validation set.
        :param val_uncertainty: The uncertainties for the validation set.
        :param test_predictions: The predicted labels for the test set.
        :param test_targets: The targets (true labels) for the test set.
        :param test_uncertainty: The uncertainties for the test set.
        :param args: The command line arguments.
        """
        f = open(args.save_uncertainty, 'w+')

        val_data = UncertaintyEvaluator._log(val_predictions,
                                             val_targets,
                                             val_uncertainty,
                                             args)
        test_data = UncertaintyEvaluator._log(test_predictions,
                                              test_targets,
                                              test_uncertainty,
                                              args)

        json.dump({'validation': val_data, 'test': test_data}, f)
        f.close()

    @staticmethod
    def _log(predictions: np.ndarray,
             targets: np.ndarray,
             uncertainty: np.ndarray,
             args: Namespace) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
        """
        Takes a collection of predictions and returns a sorted log.

        :param predictions: The predicted labels for the collection.
        :param targets: The targets (true labels) for the collection.
        :param uncertainty: The uncertainties for the collection.
        :param args: The command line arguments.

        :return: The uncertainty estimator's predictions sorted by
                 uncertainty and true error.
        """
        log = {}

        # Loop through all subtasks.
        for task in range(args.num_tasks):
            mask = targets[:, task] != None

            task_predictions = np.extract(mask, predictions[:, task])
            task_targets = np.extract(mask, targets[:, task])
            task_uncertainty = np.extract(mask, uncertainty[:, task])
            task_error = list(task_predictions - task_targets)

            task_sets = [{'prediction': task_set[0],
                          'target': task_set[1],
                          'uncertainty': task_set[2],
                          'error': task_set[3]} for task_set in zip(
                                        task_predictions,
                                        task_targets,
                                        task_uncertainty,
                                        task_error)]

            sets_by_uncertainty = sorted(task_sets,
                                         key=lambda pair: pair['uncertainty'],
                                         reverse=True)

            sets_by_error = sorted(task_sets,
                                   key=lambda pair: np.abs(pair['error']),
                                   reverse=True)

            log[args.task_names[task]] = {
                'sets_by_uncertainty': sets_by_uncertainty,
                'sets_by_error': sets_by_error}

        return log

    @staticmethod
    def visualize(file_path: str, methods: List[str]):
        """
        Visualizes stored predictions of an uncertainty estimator.

        :param file_path: The file which stores the prediction log.
        :param methods: The methods to use for evaluation and visualization.
        """
        f = open(file_path)
        log = json.load(f)['test']

        for task, data in log.items():
            for method in UncertaintyEvaluator.methods:
                if method.name in methods:
                    method.visualize(task, data)

        f.close()

    @staticmethod
    def evaluate(file_path: str, methods: List[str]) -> Dict[str, Any]:
        """
        Evaluates stored predictions of an uncertainty estimator.

        :param file_path: The file which stores the prediction log.
        :param methods: The methods to use for evaluation.
        :returns: A dictionary of evaluation results.
        """
        f = open(file_path)
        log = json.load(f)['test']

        all_evaluations = {}
        for task, data in log.items():
            task_evaluations = {}
            for method in UncertaintyEvaluator.methods:
                if method.name in methods:
                    task_evaluations[method.name] = method.evaluate(data)
            all_evaluations[task] = task_evaluations

        f.close()

        return all_evaluations

    @staticmethod
    def calibrate(lambdas: List[Callable],
                  beta_init: List[float],
                  file_path: str):
        """
        Given the stored predictions of an uncertainty estimator and fixed
        transformations, performs a calibration by selecting an optimal
        weighting of the transformations to minimize the NLL of producing
        the observed errors.

        For example, to perform a linear calibration, two lambdas might be
        passed:

        [lambda x: x, lambda x: 1]

        With initial weights of [1, 0] (the identity function).

        :param lambdas: A list of fixed transformations to apply to the
                        uncalibrated uncertainty estimates.
        :param beta_init: A list that defines the initial weighting of the
                          transformations, before optimization.
        :param file_path: The file which stores the prediction log.
        """
        def objective_function(beta: List[float],
                               uncertainty: List[float],
                               errors: List[float],
                               lambdas: List[Callable]) -> float:
            """
            Defines the cost imposed (NLL) by a particular calibration.

            :param beta: The transformation weights used in calibration.
            :param uncertainty: A list of uncalibrated uncertainty estimates.
            :param errors: The list of true prediction erros.
            :param lambdas: The list of transformations used in calibration.

            :return: The NLL of producing the observed errors given the
                     calibrated uncertainties.
            """
            # Construct prediction through lambdas and betas.
            pred_vars = np.zeros(len(uncertainty))

            for i in range(len(beta)):
                pred_vars += np.abs(beta[i]) * lambdas[i](uncertainty**2)
            pred_vars = np.clip(pred_vars, 0.001, None)
            costs = np.log(pred_vars) / 2 + errors**2 / (2 * pred_vars)

            return(np.sum(costs))

        def calibrate_sets(sets: List[Dict[str, float]],
                           sigmas: List[float],
                           lambdas: List[Callable]) -> List[Dict[str, float]]:
            """
            Calibrates a collection of uncertainty estimates.

            :param sets: An UncertaintyEvaluator log.
            :param sigmas: Optimized transformation weights (betas).
            :param lambdas: The list of transformations used in calibration.
            :return: The UncertaintyEvaluator log with calibrated
                     uncertainties.
            """
            calibrated_sets = []
            for set_ in sets:
                calibrated_set = set_.copy()
                calibrated_set['uncertainty'] = 0

                for i in range(len(sigmas)):
                    uncertainty = lambdas[i](set_['uncertainty']**2)
                    calibrated_set['uncertainty'] += sigmas[i] * uncertainty
                calibrated_sets.append(calibrated_set)
            return calibrated_sets

        f = open(file_path)
        full_log = json.load(f)
        val_log = full_log['validation']
        test_log = full_log['test']

        scaled_val_log = {}
        scaled_test_log = {}

        calibration_coefficients = {}
        for task in val_log:
            # Sample from validation data.
            samples = val_log[task]['sets_by_error']

            # Calibrate based on sampled data.
            uncertainty = np.array([set_['uncertainty'] for set_ in samples])
            errors = np.array([set_['error'] for set_ in samples])

            result = minimize(objective_function,
                              beta_init,
                              args=(uncertainty, errors, lambdas),
                              method='BFGS',
                              options={'maxiter': 500})

            calibration_coefficients[task] = np.abs(result.x)

            scaled_val_data = {}
            scaled_val_data['sets_by_error'] = calibrate_sets(
                val_log[task]['sets_by_error'], np.abs(result.x), lambdas)
            scaled_val_data['sets_by_uncertainty'] = calibrate_sets(
                val_log[task]['sets_by_uncertainty'],
                np.abs(result.x),
                lambdas)
            scaled_val_log[task] = scaled_val_data

            scaled_test_data = {}
            scaled_test_data['sets_by_error'] = calibrate_sets(
                test_log[task]['sets_by_error'], np.abs(result.x), lambdas)
            scaled_test_data['sets_by_uncertainty'] = calibrate_sets(
                test_log[task]['sets_by_uncertainty'],
                np.abs(result.x),
                lambdas)
            scaled_test_log[task] = scaled_test_data

        f.close()

        return {'validation': scaled_val_log,
                'test': scaled_test_log}, calibration_coefficients
