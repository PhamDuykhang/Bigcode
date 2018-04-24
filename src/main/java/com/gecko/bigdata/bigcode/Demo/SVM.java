package com.gecko.bigdata.bigcode.Demo;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
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
import scala.Tuple3;

public class SVM {
	public static void main(String[] args) {

		SparkConf conf = new SparkConf().setAppName("SVMvsNavie Bayes").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		String path = "D:/SVMTrain/Input/*";
		
		JavaRDD<LabeledPoint> training = sc.textFile(path).cache().map(v1 -> {
			double label = Double.parseDouble(v1.substring(0, v1.indexOf(",")));
			String featureString[] = v1.substring(v1.indexOf(":") + 1).trim().split(" ");
			double[] v = new double[featureString.length];
			int i = 0;
			for (String s : featureString) {
				if (s.trim().equals(""))
					continue;
				v[i++] = Double.parseDouble(s.trim());
			}
			return new LabeledPoint(label, Vectors.dense(v));
		});
		training.saveAsTextFile("D:/SVMTrain/train");
		System.out.println(training.count());
		String pathTest = "D:/SVMTrain/inputtest/*";

		JavaRDD<Tuple2<String, LabeledPoint>> test = sc.textFile(pathTest).cache().map(v1 -> {
			double label = Double.parseDouble(v1.substring(0, v1.indexOf(",")));
			String fileName = v1.substring(v1.indexOf(",") + 1, v1.indexOf(":")).trim();
			String featureString[] = v1.substring(v1.indexOf(":") + 1).trim().split(" ");
			double[] v = new double[featureString.length];
			int i = 0;
			for (String s : featureString) {
				if (s.trim().equals(""))
					continue;
				v[i++] = Double.parseDouble(s.trim());
			}
			return new Tuple2<String, LabeledPoint>(fileName, new LabeledPoint(label, Vectors.dense(v)));
		});
		test.saveAsTextFile("D:/SVMTrain/Test11");
		System.out.println(test.count());
		String iterations = "70";
		final SVMModel svmModel = SVMWithSGD.train(training.rdd(), Integer.parseInt(iterations));

		JavaRDD<Tuple3<String, Double, Double>> ResuleSVM = test.map(f -> {
			return new Tuple3<String, Double, Double>(f._1, f._2.label(), svmModel.predict(f._2.features()));
		});

		double accuracySVM = 1.0 * ResuleSVM.filter(pl -> {
			String lableName = "";
			lableName = (pl._3() == 0.0) ? "Male" : "Female";
			System.out.println(pl._1() + " -- " + pl._2() + "----" + lableName);
			return pl._2().intValue() == pl._3().intValue();
		}).count() / (double) test.count();

		System.out.println("svm accuracy : " + accuracySVM * 100 + " %");

	}
}
