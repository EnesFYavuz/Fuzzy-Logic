package paket;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;


public class Program {

	public static void main(String[] args) throws IOException {
		try (// TODO Auto-generated method stub
		Scanner in = new Scanner(System.in)) {
			int araKatmanNoron;
			double momentum,ogrenmeKatsayisi,hata;
			int epoch,sec=0;
			YSA ysa = null;
			YSA ysa1 = null;
			final XYSeries firefox = new XYSeries("Epoch Hatalari");
			final XYSeriesCollection dataset = new XYSeriesCollection();
		      dataset.addSeries(firefox);
			do {
				System.out.println("1.Momentumlu Hata Sonuclari");
				System.out.println("2.Momentumlu Tekli Test");
				System.out.println("3.Momentumsuz Hata Sonuclari");
				System.out.println("4.Epoch Hatalari");
				System.out.println("5.cikis");
				System.out.print("=>");
				sec=in.nextInt();
				switch (sec) {
				case 1:
					System.out.print("Ara katman noron sayisi:");
					araKatmanNoron = in.nextInt();
					System.out.print("Momentum:");
					momentum = in.nextDouble();
					System.out.print("Ogrenme Katsayisi:");
					ogrenmeKatsayisi = in.nextDouble();
					System.out.print("Minumum hata:");
					hata = in.nextDouble();
					System.out.print("Epoch Sayisi:");
					epoch = in.nextInt();
					ysa = new YSA(araKatmanNoron, momentum, ogrenmeKatsayisi, hata, epoch);
					ysa.egit("Momentumlu");
					System.out.println("Egitimde elde edilen hata orani"+ysa.egitimHata("Momentumlu"));
					System.out.println("Teste elde edilen hata orani:"+ysa.test());
					break;
				case 2:
					if(ysa == null) {
						System.out.println("once egitim yapilmali");
						System.in.read();
						break;
					}
					double []inputs = new double[3];
					System.out.println("ortam isik yogunlugu");
					inputs[0] = in.nextDouble();
					System.out.println("acik renk yogunlugu");
					inputs[1] = in.nextDouble();
					System.out.println("kapali renk yogunlugu");
					inputs[2] = in.nextDouble();
					double cikti =ysa.tekTest(inputs);
					System.out.println("parlaklik:"+cikti+" cd/m2");
					break;

				case 3:
					System.out.print("Ara katman noron sayisi:");
					araKatmanNoron = in.nextInt();
					System.out.print("Ogrenme Katsayisi:");
					ogrenmeKatsayisi = in.nextDouble();
					System.out.print("Minumum hata:");
					hata = in.nextDouble();
					System.out.print("Epoch Sayisi:");
					epoch = in.nextInt();
					ysa1 = new YSA(araKatmanNoron,ogrenmeKatsayisi, hata, epoch);
					ysa1.egit("Momentumsuz");
					System.out.println("Egitimde elde edilen hata orani:"+ysa1.egitimHata("Momentumsuz"));
					System.out.println("Teste elde edilen hata orani:"+ysa1.test());
				break;
				case 4:
					double []hatalar=ysa1.hatalar();
					int epoch1=1;
					for(double h:hatalar) {
						System.out.println(epoch1+":"+h);
					//	   firefox.add(epoch1,h);
						epoch1++;
					}
					JFreeChart pieChart = ChartFactory.createXYLineChart("Egitim Hata Grafigi", "Epoch Sayisi", "Hatalar",dataset,PlotOrientation.VERTICAL, true, true, false);
					
					  int width = 800;   
					  int height = 600;  
					  File lineChart = new File( "LineChart.jpeg" ); 
					  ChartUtilities.saveChartAsJPEG(lineChart ,pieChart, width ,height);
				break;

			}
			} while (sec!=5);
		}
	}

}
