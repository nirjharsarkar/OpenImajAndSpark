package com.image.analysis;

import java.util.List;

import org.apache.poi.util.SystemOutLogger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.VectorTransformer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;

/**
 * This exercise was about understanding the concepts of unsupervised learning.
 * From the previously generated image vector datas.txt, took a subset of the
 * values and took out the class identifier and created a set of image vectors
 * as in kmeans_train_data.txt. kmeans_test_data.txt has sample test image vectors. 
 * 
 * 
 *
 * 
 * @author/modifier
 * 	nirjhar.sarkar@gmail.com
 * 	https://in.linkedin.com/in/nirjharsarkar
 * 	https://twitter.com/nirjharsarkar
 * 
 *
 *
 */
public class KMeansClusteringOnImageVector {

	private static JavaSparkContext sc;

	public static void main(String[] args) {

		SparkConf conf = new SparkConf().setAppName("KMeansClusteringOnImageVector").setMaster("local");
		sc = new JavaSparkContext(conf);

		// Load and parse data
		String path = "data/mllib/kmeans_train_data.txt";
		JavaRDD<String> data = sc.textFile(path);
		JavaRDD<Vector> parsedData = data.map(new Function<String, Vector>() {
			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public Vector call(String s) {
				String[] sarray = s.split(" ");
				double[] values = new double[sarray.length];
				for (int i = 0; i < sarray.length; i++) {
					// System.out.println(sarray[i]);
					values[i] = Double.parseDouble(sarray[i]);
				}

				return Vectors.dense(values);
			}
		});
		parsedData.cache();

		// Cluster the data into two classes using KMeans
		int numClusters = 3;
		int numIterations = 99;
		KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

		// Test Data
		JavaRDD<String> testData = sc.textFile("data/mllib/kmeans_test_data.txt");
		JavaRDD<Vector> parsedTestData = testData.map(new Function<String, Vector>() {
			/**
			 * 
			 */
			private static final long serialVersionUID = -2946376955448026722L;

			public Vector call(String s) {
				String[] sarray = s.split(" ");
				double[] values = new double[sarray.length];
				for (int i = 0; i < sarray.length; i++) {
					// System.out.println(sarray[i]);
					values[i] = Double.parseDouble(sarray[i]);
				}

				return Vectors.dense(values);
			}
		});

		// Evaluate clustering by computing Within Set Sum of Squared Errors
		double WSSSE = clusters.computeCost(parsedData.rdd());
		System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
		System.out.println(clusters.toPMML());

		JavaRDD<Integer> predictedCluster = clusters.predict(parsedTestData);

		List<Integer> collections = predictedCluster.collect();

		for (Integer collectionItem : collections) {
			System.out.println(collectionItem);

		}

		// Save and load model
		/*
		 * clusters.save(sc.sc(), "myModelPath"); KMeansModel sameModel =
		 * KMeansModel.load(sc.sc(), "myModelPath");
		 */
	}

}
