package siddhesh.sparkl;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author Siddhesh Rane
 */
public class SysPropTest {

    /**
     * Check whether system properties are delivered to all worker nodes or just
     * the client
     *
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        SparkConf conf = new SparkConf().setMaster("local");
        SparkSession spark = SparkSession.builder()
//                .config(conf)
                .appName("SysPropTest")
                .getOrCreate();
        spark.range(4).repartition(4).map(SysPropTest::prop, Encoders.STRING()).show(false);
    }

    static String prop(long i) {
        boolean property = Boolean.getBoolean("nlp.opennlp");
        String nlp = System.getProperty("nlp.opennlp");
        String sfu = System.getProperty("sfu");
        return i + ")" + property + " "+nlp+" sfu="+sfu;
    }
}
