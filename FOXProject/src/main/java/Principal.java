package com.suelenfenali.foxproject.controle;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.net.MalformedURLException;
import java.util.ArrayList;
import org.aksw.fox.binding.java.FoxApi;
import org.aksw.fox.binding.java.FoxParameter;
import org.aksw.fox.binding.java.FoxResponse;
import org.aksw.fox.binding.java.IFoxApi;

/**
 *
 * @author suelenfenali
 */
public class Principal {

    static IFoxApi fox = new FoxApi();
    static ArrayList<String> resultados = new ArrayList<>();
    static BufferedReader lineReader;
    static BufferedWriter outputWriter;
    static BufferedWriter statisticsWriter;
    static ArrayList<Integer> numberEntitiesPerTweet = new ArrayList<>();
    static double totalNumberOfEntities = 0;
    static double totalTweetsProcessed = 0;
    static double totalTweetsWithEntity = 0;
    static double avgEntitiesPerTweet = 0;
    static double avgEntitiesPerAnnotatedTweet = 0;

    public static void main(String[] args) throws Exception {
        String line;
        
        testeFox();
        
       lineReader = new BufferedReader(new FileReader("data/tweetsEnglish_filtered.txt"));
       outputWriter = new BufferedWriter(new FileWriter("data/output_filtered.txt"));
       statisticsWriter = new BufferedWriter(new FileWriter("data/statistics_filtered.txt"));

       line = lineReader.readLine();
       while (line != null) {
           sendToFox(line);
           line = lineReader.readLine();
           totalTweetsProcessed++;
       }
       avgEntitiesPerTweet = totalNumberOfEntities/totalTweetsProcessed;
       avgEntitiesPerAnnotatedTweet = totalNumberOfEntities/totalTweetsWithEntity;
       System.out.println(avgEntitiesPerTweet);
       System.out.println(avgEntitiesPerAnnotatedTweet);
       statisticsWriter.append("Total de tweets processados: " + totalTweetsProcessed);
       statisticsWriter.newLine();
       statisticsWriter.append("Total de tweets anotados: " + totalTweetsWithEntity);
       statisticsWriter.newLine();
       statisticsWriter.append("Total de anotações realizadas: " + totalNumberOfEntities);
       statisticsWriter.newLine();
       statisticsWriter.append("Média de anotações por tweet processado: " + avgEntitiesPerTweet);
       statisticsWriter.newLine();
       statisticsWriter.append("Média de anotações por tweet anotado: " + avgEntitiesPerAnnotatedTweet);
       
       outputWriter.close();
       statisticsWriter.close();
    }

    public static void sendToFox(String tweet) throws MalformedURLException, IOException {

        fox.setInput(tweet);
        fox.setOutputFormat(FoxParameter.OUTPUT.JSONLD);
        FoxResponse response = fox.send();
        outputWriter.append(response.getOutput());
        outputWriter.newLine();
        System.out.println(tweet);
        System.out.println(response.getLog());
        calcMetrics(response.getLog());
    }
    

    public static void calcMetrics(String log) throws IOException {
        BufferedReader logReader = new BufferedReader(new StringReader(log));
        String logLine;
        String caracter = "";
        String foundNumber;
        int foundEntities = 0;
        int numberLines = 1;
        logLine = logReader.readLine();
        while (logLine != null && numberLines < 10) {
            caracter = Character.toString(logLine.charAt(11));
            if (caracter.equals("F")) {
                foundNumber = logLine.substring(17, 18);
                foundNumber = foundNumber.trim();
                foundEntities = Integer.parseInt(foundNumber);
                numberEntitiesPerTweet.add(foundEntities);
                totalNumberOfEntities = totalNumberOfEntities + foundEntities;
                if (foundEntities > 0) {
                    totalTweetsWithEntity = totalTweetsWithEntity + 1;
                }
                break;
            }
            logLine = logReader.readLine();
            numberLines = numberLines + 1;
        }
    }

}
