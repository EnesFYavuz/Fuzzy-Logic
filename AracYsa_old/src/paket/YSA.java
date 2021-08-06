package paket;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Scanner;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;


public class YSA {
	private static final File egitimDosya = new File(YSA.class.getResource("Egitim.txt").getPath());
	private static final File testDosya = new File(YSA.class.getResource("Test.txt").getPath());
	private double[] maksimumlar;
	private double[] minumumlar;
	
	private DataSet egitimVeriSeti;
	private DataSet testVeriSeti;
	private int araKatmanNoronSayisi;
	MomentumBackpropagation bp;
	BackPropagation momentumsuzbp;
	double epoch;
	double []eldeEdilenHatalar;
	public YSA(int araKatmanNoronSayisi,double ogrenmeKatsayisi,double hata,int epoch2) throws FileNotFoundException {
		// TODO Auto-generated constructor stub
		maksimumlar = new double[4];
		minumumlar = new double[4];
		
		for(int i=0;i<4;i++) {
			maksimumlar[i] = Double.MIN_VALUE;
			minumumlar[i] = Double.MAX_VALUE;
		}
		VeriSetiMaks(egitimDosya);
		VeriSetiMaks(testDosya);
		egitimVeriSeti = VeriSeti(egitimDosya);
		testVeriSeti = VeriSeti(testDosya);
		momentumsuzbp = new BackPropagation();
		momentumsuzbp.setLearningRate(ogrenmeKatsayisi);
		momentumsuzbp.setMaxError(hata);
		this.epoch=epoch2;
		this.araKatmanNoronSayisi = araKatmanNoronSayisi;
		eldeEdilenHatalar = new double [(int) epoch];
	}
	public YSA(int araKatmanNoronSayisi,double momentum,double ogrenmeKatsayisi,double hata,int epoch) throws FileNotFoundException {
		maksimumlar = new double[4];
		minumumlar = new double[4];
		
		for(int i=0;i<4;i++) {
			maksimumlar[i] = Double.MIN_VALUE;
			minumumlar[i] = Double.MAX_VALUE;
		}
		VeriSetiMaks(egitimDosya);
		VeriSetiMaks(testDosya);
		egitimVeriSeti = VeriSeti(egitimDosya);
		testVeriSeti = VeriSeti(testDosya);
		
		bp = new MomentumBackpropagation();
		bp.setMomentum(momentum);
		bp.setLearningRate(ogrenmeKatsayisi);
		bp.setMaxError(hata);
		bp.setMaxIterations(epoch);
		this.araKatmanNoronSayisi = araKatmanNoronSayisi;
	}
	MultiLayerPerceptron BackprogadionChoose=null;
	public void egit(String secim) throws FileNotFoundException {
		if(secim=="Momentumlu") {
			MultiLayerPerceptron sinirselAg = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,3,araKatmanNoronSayisi,1);
			sinirselAg.setLearningRule(bp);
			sinirselAg.learn(egitimVeriSeti);
			BackprogadionChoose = sinirselAg;
		}
		else {
			MultiLayerPerceptron sinirselAg1 = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,3,araKatmanNoronSayisi,1);
			sinirselAg1.setLearningRule(momentumsuzbp);
			for(int i=0;i<epoch;i++) {
				sinirselAg1.getLearningRule().doOneLearningIteration(egitimVeriSeti);
				if(i==0) eldeEdilenHatalar[i]=1;
				else eldeEdilenHatalar[i]=sinirselAg1.getLearningRule().getPreviousEpochError();
			}
			BackprogadionChoose = sinirselAg1;
		}
		BackprogadionChoose.save("ogrenenAg.nnet");
		System.out.println("Egitim tamamlandi.");
	}
	public double[] hatalar () {
		return eldeEdilenHatalar;
	}
	public double egitimHata(String secim) {
		if(secim=="Momentumlu") {
		return bp.getTotalNetworkError();
		}
		else {
			return momentumsuzbp.getTotalNetworkError();
		}
	}
	private double mse(double[] beklenen,double[] cikti) {
		double birSatirVeriToplamHata=0;
		birSatirVeriToplamHata += Math.pow((beklenen[0]-cikti[0]), 2);
		return birSatirVeriToplamHata/3;
		
	}
	public double test() {
		NeuralNetwork sinirselAg = NeuralNetwork.createFromFile("ogrenenAg.nnet");
		double toplamHata = 0;
		List<DataSetRow> satirlar = testVeriSeti.getRows();
		for (DataSetRow dr : satirlar) {
			sinirselAg.setInput(dr.getInput());
			sinirselAg.calculate();
			toplamHata += mse(dr.getDesiredOutput(),sinirselAg.getOutput());
		}
		return toplamHata/testVeriSeti.size();
	}
	public double Sonuc(double[] outputs) {
		double maks = outputs[0] * 1500;
		return maks;
		
	}
	public double tekTest(double[] girdiler) {
	 for (int i = 0; i < 3; i++) {
		 girdiler[i] = minMax(maksimumlar[i], minumumlar[i], girdiler[i]);
	}
	 NeuralNetwork sinirselAg = NeuralNetwork.createFromFile("ogrenenAg.nnet");
	 sinirselAg.setInput(girdiler);
	 sinirselAg.calculate();
	 return Sonuc(sinirselAg.getOutput());
	}
	public void VeriSetiMaks(File file) throws FileNotFoundException {
		Scanner fl = new Scanner(file);
		while(fl.hasNextDouble()) {
			for(int i=0;i<4;i++) {
				double d = fl.nextDouble();
				if(d > maksimumlar[i]) maksimumlar[i]=d;
				if(d < minumumlar[i]) minumumlar[i]=d;
			}
		}
		fl.close();
	}
	private double minMax(double max,double min,double x) {
		return (x-min)/(max-min);
	}
	private DataSet VeriSeti(File file) throws FileNotFoundException {
		Scanner fl = new Scanner(file);
		DataSet ds = new DataSet(3,1);
		while(fl.hasNextDouble()) {
			double []inputs = new double[3];
			double []output = new double[1];
			for(int i=0;i<3;i++) {
				double d = fl.nextDouble();
				inputs[i] = minMax(maksimumlar[i],minumumlar[i],d);
			}
			output[0] = minMax(maksimumlar[3],minumumlar[3],fl.nextDouble());
			ds.add(new DataSetRow(inputs,output));
			
		}
		fl.close();
		return ds;
	}
	
	
	
	
}
