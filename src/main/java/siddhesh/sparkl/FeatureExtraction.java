package siddhesh.sparkl;

import java.util.Arrays;
import java.util.stream.Collectors;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.udf;
import org.apache.spark.sql.types.DataTypes;
import scala.collection.mutable.WrappedArray;

/**
 *
 * @author Siddhesh Rane
 */
public class FeatureExtraction {

    public static void main(String[] args) {
        FeatureExtraction fe = new FeatureExtraction();
        fe.loadFromCsv(args);
    }

    private final SparkSession spark;

    public FeatureExtraction() {
        spark = SparkSession.builder()
                //                .config(conf)
                .appName("Sid2Sparkl")
                .getOrCreate();
    }
    public FeatureExtraction(SparkSession spark) {
        this.spark = spark;
    }

    public void loadFromCsv(String... path) {
        Dataset<Row> csv = spark.read()
                .option("sep", "\t")
                .option("header", "true")
                .csv(path);

        UserDefinedFunction str2arr = udf((String sent) -> sent == null ? new String[]{} : sent.split(" "), DataTypes.createArrayType(DataTypes.StringType));
        UserDefinedFunction arr2set = udf((WrappedArray<String> arr) -> arr.toSet().toList(), DataTypes.createArrayType(DataTypes.StringType));

        csv = csv
                .withColumn("left", str2arr.apply(col("leftcsv")))
                .withColumn("right", str2arr.apply(col("rightcsv")))
                //        csv
                .withColumn("leftset", arr2set.apply(col("left")))
                .withColumn("rightset", arr2set.apply(col("right")));

        csv.sample(0.2).show();
        FPGrowthModel fpleft = new FPGrowth()
                .setItemsCol("leftset")
                .fit(csv);
        FPGrowthModel fpright = new FPGrowth()
                .setItemsCol("rightset")
                .fit(csv);
        
        fpleft.associationRules().show(false);
        fpright.associationRules().show();
        
        fpleft.freqItemsets().show();
        fpright.freqItemsets().show();
    }
    public void countVectorize(Dataset d) {
        CountVectorizerModel leftCounts = //                    .setMinDF(2)
        new CountVectorizer().setInputCol("left").setOutputCol("leftCount").setVocabSize(100).fit(d);
        String[] leftvocab = leftCounts.vocabulary();
        StopWordsRemover removeStopWordsL = new StopWordsRemover().setInputCol("left").setOutputCol("filteredLeft").setStopWords(Arrays.copyOfRange(leftvocab, (int) (leftvocab.length * 0.7), leftvocab.length));
        CountVectorizerModel rightCounts = //                    .setMinDF(2)
        new CountVectorizer().setInputCol("right").setOutputCol("rightCount").setVocabSize(100).fit(d);
        String[] rightvocab = rightCounts.vocabulary();
        StopWordsRemover removeStopWordsR = new StopWordsRemover().setInputCol("right").setOutputCol("filteredRight").setStopWords(Arrays.copyOfRange(rightvocab, (int) (rightvocab.length * 0.7), rightvocab.length));
        System.out.println("Left Vocabsize:" + leftvocab.length);
        System.out.println(Arrays.stream(leftvocab).limit(100).collect(Collectors.joining("|")));
        System.out.println("Right Vocabsize:" + rightvocab.length);
        System.out.println(Arrays.stream(rightvocab).limit(100).collect(Collectors.joining("|")));
        removeStopWordsL.transform(d).dropDuplicates("entity").show();
        removeStopWordsR.transform(d).dropDuplicates("entity").show();
    }
    public void fpGrowth(Dataset d) {
        FPGrowthModel fp = new FPGrowth().setItemsCol("left").setMinSupport(0.02).setMinConfidence(0.2).fit(d);
        fp.freqItemsets().show(false);
        fp.associationRules().show(false);
    }
}
