package com.image.analysis;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;


/**
 * 
 * @modifier
 * 	nirjhar.sarkar@gmail.com
 * 	https://in.linkedin.com/in/nirjharsarkar
 * 	https://twitter.com/nirjharsarkar
 * 
 *
 */

public class NavieBayesClassification {
	private static final String DATA = "C:/Nirjhar Data/Developer Zone/imageanalysis/BagOfVisualWords/datas.txt";
	private static final String TEST_DATA = "C:/Nirjhar Data/Developer Zone/imageanalysis/BagOfVisualWords/Test.csv";
	private static JavaSparkContext sc;

	@SuppressWarnings("serial")
	public static void main(String[] args) {

		// Create Java spark context
		SparkConf conf = new SparkConf().setAppName("SVM vs Navie Bayes").setMaster("local").set("spark.executor.memory", "2g").set("spark.driver.memory", "4g");;
		sc = new JavaSparkContext(conf);

		// RDD training = MLUtils.loadLabeledData(sc, args[0]);
		// RDD test = MLUtils.loadLabeledData(sc, args[1]); // test set
		
		

		JavaRDD<LabeledPoint> input = sc
				.textFile(DATA).cache()
				.map(new Function<String, LabeledPoint>() {

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
		
		JavaRDD<LabeledPoint>[] tmp = input.randomSplit(new double[]{0.6, 0.4}, 12345);
		
		JavaRDD<LabeledPoint> training = tmp[0]; // training set
	    JavaRDD<LabeledPoint> test = tmp[1]; // test set
	    
	    training.collect();
	    test.collect();
		
		System.out.println(training.count());
		
		/*JavaRDD<LabeledPoint> test = sc
				.textFile(TEST_DATA).cache()
				.map(new Function<String, LabeledPoint>() {

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
		*/
		System.out.println(test.count());
		
		final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 115.0, "multinomial");

		JavaPairRDD<Double, Double> predictionAndLabel = test
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {

					private static final long serialVersionUID = 1L;

					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
					}
				});
		
		
		double accuracy = 1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {

			public Boolean call(Tuple2<Double, Double> pl) {
				return pl._1().intValue() == pl._2().intValue();
			}
		}).count() / (double) test.count();
		
		System.out.println("navie bayes accuracy : " + accuracy);

		
	}
}