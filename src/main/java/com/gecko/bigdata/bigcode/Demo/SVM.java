package com.gecko.bigdata.bigcode.Demo;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
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

		SparkConf conf = new SparkConf().setAppName("SVMvs").setMaster("local");
//		SparkContext Spc = new SparkContext(conf);
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
		
		String iterations = "300";
		final SVMModel svmModel = SVMWithSGD.train(training.rdd(), Integer.parseInt(iterations));
//		svmModel.save(Spc,"D:/SVMTrain/SVMModel");
		JavaRDD<Tuple3<String, Double, Double>> ResultSVM = test.map(f -> {
			return new Tuple3<String, Double, Double>(f._1, f._2.label(), svmModel.predict(f._2.features()));
		});
		
		ResultSVM.foreach(line->{
			String fLable="";
			String lLable="";
			fLable=(line._2() == 0.0) ? "Male" : "Female";
			lLable=(line._3() == 0.0) ? "Male" : "Female";
			System.out.println(line._1()+"--"+fLable+"----"+lLable);
		});
		double accuracySVM = 1.0 * ResultSVM.filter(pl -> {
			return pl._2().intValue() == pl._3().intValue();
		}).count() / (double) test.count();

		System.out.println("svm accuracy : " + accuracySVM * 100 + " %");

	}
}
