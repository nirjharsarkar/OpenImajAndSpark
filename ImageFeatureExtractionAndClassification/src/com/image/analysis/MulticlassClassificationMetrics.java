package com.image.analysis;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
// $example off$
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

/**
 * Limited Memory Broyden–Fletcher–Goldfarb–Shanno algorithm
 * 
 * @author sarkan
 *
 */
public class MulticlassClassificationMetrics {

	private static final String DATA = "C:/Nirjhar Data/Developer Zone/imageanalysis/BagOfVisualWords/datas.txt";
	private static JavaSparkContext sc;

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("SVM vs Navie Bayes").setMaster("local")
				.set("spark.executor.memory", "2g").set("spark.driver.memory", "4g");
		;
		sc = new JavaSparkContext(conf);



		JavaRDD<LabeledPoint> input = sc.textFile(DATA).cache().map(new Function<String, LabeledPoint>() {

			private static final long serialVersionUID = 1L;

			public LabeledPoint call(String v1) throws Exception {
				double label = Double.parseDouble(v1.substring(0, v1.indexOf(",")));
				String temp = v1.substring(v1.indexOf(",") + 1).trim();
				String featureString[] = temp.substring(temp.indexOf(",") + 1).trim().split(" ");
				double[] v = new double[featureString.length];
				int i = 0;
				for (String s : featureString) {
					if (s.trim().equals(""))
						continue;
					v[i++] = Double.parseDouble(s.trim());
				}
				return new LabeledPoint(label, Vectors.dense(v));
			}

		});

		// Split initial RDD into two... [60% training data, 40% testing data].
		JavaRDD<LabeledPoint>[] splits = input.randomSplit(new double[] { 0.6, 0.4 }, 11L);
		JavaRDD<LabeledPoint> training = splits[0].cache();
		JavaRDD<LabeledPoint> test = splits[1];

		// Run training algorithm to build the model.
		final LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(97).setFeatureScaling(true).run(training.rdd());

		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test
				.map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double prediction = model.predict(p.features());
						return new Tuple2<Object, Object>(prediction, p.label());
					}
				});

		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n" + confusion);

		/*
		 * In pattern recognition and information retrieval with binary
		 * classification, precision (also called positive predictive value) is
		 * the fraction of retrieved instances that are relevant, while recall
		 * (also known as sensitivity) is the fraction of relevant instances
		 * that are retrieved. Both precision and recall are therefore based on
		 * an understanding and measure of relevance. Suppose a computer program
		 * for recognizing dogs in scenes from a video identifies 7 dogs in a
		 * scene containing 9 dogs and some cats. If 4 of the identifications
		 * are correct, but 3 are actually cats, the program's precision is 4/7
		 * while its recall is 4/9. When a search engine returns 30 pages only
		 * 20 of which were relevant while failing to return 40 additional
		 * relevant pages, its precision is 20/30 = 2/3 while its recall is
		 * 20/60 = 1/3.
		 */
		//So, in this case, precision is "how useful the search results are", and recall is "how complete the results are".
		// Overall statistics
		System.out.println("Precision = " + metrics.precision());
		System.out.println("Recall = " + metrics.recall());
		/*
		 * In statistical analysis of binary classification, the F1 score (also
		 * F-score or F-measure) is a measure of a test's accuracy. It considers
		 * both the precision p and the recall r of the test to compute the
		 * score: p is the number of correct positive results divided by the
		 * number of all positive results, and r is the number of correct
		 * positive results divided by the number of positive results that
		 * should have been returned. The F1 score can be interpreted as a
		 * weighted average of the precision and recall, where an F1 score
		 * reaches its best value at 1 and worst at 0. The traditional F-measure
		 * or balanced F-score (F1 score) is the harmonic mean of precision and
		 * recall: F1 = 2*(precision.recall)/(precision+recall)
		 */
		System.out.println("F1 Score = " + metrics.fMeasure());

		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
			System.out.format("Class %f precision = %f\n", metrics.labels()[i], metrics.precision(metrics.labels()[i]));
			System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(metrics.labels()[i]));
			System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(metrics.labels()[i]));
		}

		// Weighted stats
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());

		// Save and load model
		model.save(JavaSparkContext.toSparkContext(sc), "target/tmp/LogisticRegressionModel");
		LogisticRegressionModel sameModel = LogisticRegressionModel.load(JavaSparkContext.toSparkContext(sc),
				"target/tmp/LogisticRegressionModel");
		// $example off$

	}

}
