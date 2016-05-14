package com.image.analysis;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.openimaj.data.DataSource;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.SparseIntFVComparison;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BagOfVisualWordsFeatureExtraction {

	/**
	 * 
	 * @param args
	 *            ignored
	 * @throws IOException
	 *             if the image can't be read
	 */

	private static Logger slf4jLogger = LoggerFactory.getLogger(BagOfVisualWordsFeatureExtraction.class);
	private static List<File> fileNames = new ArrayList<File>();
	private static final String TYPE_STR = "FullRun";

	private static String FOLDER_PATH = "C:/Nirjhar Data/Developer Zone/imageanalysis/101_ObjectCategories";
	private static String CSV_PATH = "C:/Nirjhar Data/Developer Zone/imageanalysis/BagOfVisualWords/" + TYPE_STR+ ".txt";
	
	private static List<String> imageClassArray = new ArrayList<String>();

	public static void main(String[] args) throws IOException {

		addImageClass();

		/*
		 * File[] imageURLs = new File[] { new File("images/alebes.1.jpg"), new
		 * File("images/alebes.2.jpg"), new File("images/amflem.2.jpg") };
		 */

		List<File> imageURLs = getFiles(new File(FOLDER_PATH).listFiles());



		// Create an engine to extract some local features; in this case, we'll
		// get SIFT features located at the extrema in the
		// difference-of-Gaussian pyramid.
		final DoGSIFTEngine engine = new DoGSIFTEngine();

		// Now we load some load some images and extract their features. As
		// we're going to reuse these, we'll store them in a map of the image
		// identifier to list of local features.
		final Map<URL, LocalFeatureList<Keypoint>> imageKeypoints = new HashMap<URL, LocalFeatureList<Keypoint>>();
		for (final File url : imageURLs) {
			// load image

			// System.out.println(url.getParent());
			final FImage image = ImageUtilities.readF(url);

			// extract the features and store them in the map against the image
			// url
			imageKeypoints.put(url.toURI().toURL(), engine.findFeatures(image));
		}

		// Next we need to cluster the features to build the set of visual
		// terms. We'll setup the clustering algorithm to create 200 visual
		// terms using approximate k-means.
		final ByteKMeans kmeans = ByteKMeans.createKDTreeEnsemble(200);

		// We need to get the data in the correct format for the clustering.
		// This can be done manually by copying the raw feature data into an
		// array, or by creating a DataSource as shown:
		final DataSource<byte[]> datasource = new LocalFeatureListDataSource<Keypoint, byte[]>(imageKeypoints);

		// Then we use the DataSource as input to the clusterer and get the
		// resultant centroids
		final ByteCentroidsResult result = kmeans.cluster(datasource);

		// In this example we want to create a standard BoVW model which uses
		// hard-assignment; this means that each local feature is mapped to a
		// single visual word. We can just use the default hard assigner to
		// achieve this.
		final HardAssigner<byte[], ?, ?> assigner = result.defaultHardAssigner();

		// We create a new BagOfVisualWords instance using our assigner, and
		// then use this to extract a vector representing the number of
		// occurrences of each visual word in our input images.
		final BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

		// We'll store the resultant occurrence vectors in a map
		final Map<URL, SparseIntFV> occurrences = new HashMap<URL, SparseIntFV>();
		for (final Entry<URL, LocalFeatureList<Keypoint>> entry : imageKeypoints.entrySet()) {
			occurrences.put(entry.getKey(), bovw.aggregate(entry.getValue()));
		}

		// That's basically it; from this point onwards you could use the
		// vectors to train a classifier, or measure the distance between them
		// to assess the similarity of the input images. To finish up, we'll
		// compute and print the distance matrix of our input images:

		

		for (final Entry<URL, SparseIntFV> entry1 : occurrences.entrySet()) {

		/*	for (final Entry<URL, SparseIntFV> entry2 : occurrences.entrySet()) {
				// this computes the Euclidean distance. Note that we're not
				// normalising the vectors here, but in reality you probably
				// would want to.
				final double distance = SparseIntFVComparison.EUCLIDEAN.compare(entry1.getValue(), entry2.getValue());

				System.out.format("%2.3f\t", distance);
			}*/

			//System.out.println();

			int[] entryVector = entry1.getValue().getVector().toArray();


			slf4jLogger.info(getImageClass(entry1.getKey().getFile()) + ","
					+ Arrays.toString(entryVector).replace("[", "").replace("]", "").replace(",", ""));
			
		}

	}
	public static List<File> getFiles(File[] files) {

		// System.out.println(files.length);

		for (File file : files) {
			if (file.isDirectory()) {
				getFiles(file.listFiles());

			} else {
				fileNames.add(new File(file.getPath()));
			}
		}

		return fileNames;

	}

	public static String getImageClass(String filepath) {

		File file = new File(filepath);

		return imageClassArray.indexOf(file.getParentFile().getName())+","+file.getName();

	}

	public static void addImageClass() {
		imageClassArray.add("accordion");
		imageClassArray.add("airplanes");
		imageClassArray.add("anchor");
		imageClassArray.add("ant");
		imageClassArray.add("BACKGROUND_Google");
		imageClassArray.add("barrel");
		imageClassArray.add("bass");
		imageClassArray.add("beaver");
		imageClassArray.add("binocular");
		imageClassArray.add("bonsai");
		imageClassArray.add("brain");
		imageClassArray.add("brontosaurus");
		imageClassArray.add("buddha");
		imageClassArray.add("butterfly");
		imageClassArray.add("camera");
		imageClassArray.add("cannon");
		imageClassArray.add("car_side");
		imageClassArray.add("ceiling_fan");
		imageClassArray.add("cellphone");
		imageClassArray.add("chair");
		imageClassArray.add("chandelier");
		imageClassArray.add("cougar_body");
		imageClassArray.add("cougar_face");
		imageClassArray.add("crab");
		imageClassArray.add("crayfish");
		imageClassArray.add("crocodile");
		imageClassArray.add("crocodile_head");
		imageClassArray.add("cup");
		imageClassArray.add("dalmatian");
		imageClassArray.add("dollar_bill");
		imageClassArray.add("dolphin");
		imageClassArray.add("dragonfly");
		imageClassArray.add("electric_guitar");
		imageClassArray.add("elephant");
		imageClassArray.add("emu");
		imageClassArray.add("euphonium");
		imageClassArray.add("ewer");
		imageClassArray.add("Faces");
		imageClassArray.add("Faces_easy");
		imageClassArray.add("ferry");
		imageClassArray.add("flamingo");
		imageClassArray.add("flamingo_head");
		imageClassArray.add("garfield");
		imageClassArray.add("gerenuk");
		imageClassArray.add("gramophone");
		imageClassArray.add("grand_piano");
		imageClassArray.add("hawksbill");
		imageClassArray.add("headphone");
		imageClassArray.add("hedgehog");
		imageClassArray.add("helicopter");
		imageClassArray.add("ibis");
		imageClassArray.add("inline_skate");
		imageClassArray.add("joshua_tree");
		imageClassArray.add("kangaroo");
		imageClassArray.add("ketch");
		imageClassArray.add("lamp");
		imageClassArray.add("laptop");
		imageClassArray.add("Leopards");
		imageClassArray.add("llama");
		imageClassArray.add("lobster");
		imageClassArray.add("lotus");
		imageClassArray.add("mandolin");
		imageClassArray.add("mayfly");
		imageClassArray.add("menorah");
		imageClassArray.add("metronome");
		imageClassArray.add("minaret");
		imageClassArray.add("Motorbikes");
		imageClassArray.add("nautilus");
		imageClassArray.add("octopus");
		imageClassArray.add("okapi");
		imageClassArray.add("pagoda");
		imageClassArray.add("panda");
		imageClassArray.add("pigeon");
		imageClassArray.add("pizza");
		imageClassArray.add("platypus");
		imageClassArray.add("pyramid");
		imageClassArray.add("revolver");
		imageClassArray.add("rhino");
		imageClassArray.add("rooster");
		imageClassArray.add("saxophone");
		imageClassArray.add("schooner");
		imageClassArray.add("scissors");
		imageClassArray.add("scorpion");
		imageClassArray.add("sea_horse");
		imageClassArray.add("snoopy");
		imageClassArray.add("soccer_ball");
		imageClassArray.add("stapler");
		imageClassArray.add("starfish");
		imageClassArray.add("stegosaurus");
		imageClassArray.add("stop_sign");
		imageClassArray.add("strawberry");
		imageClassArray.add("sunflower");
		imageClassArray.add("tick");
		imageClassArray.add("trilobite");
		imageClassArray.add("umbrella");
		imageClassArray.add("watch");
		imageClassArray.add("water_lilly");
		imageClassArray.add("wheelchair");
		imageClassArray.add("wild_cat");
		imageClassArray.add("windsor_chair");
		imageClassArray.add("wrench");
		imageClassArray.add("yin_yang");
	}
}