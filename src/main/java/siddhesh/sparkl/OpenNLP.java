package siddhesh.sparkl;

import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

/**
 * Singleton class with static methods for OpenNLP tasks like tokenization,
 * sentence splitting and pos tagging. All fields and initialization is done
 * statically, so that Spark will not have to broadcast these variables
 * everywhere.
 *
 * @author Siddhesh Rane
 */
public final class OpenNLP extends NLP {

    //Models are thread safe. share them
    public SentenceModel sentenceModel;
    public TokenizerModel tokenizerModel;
    public POSModel posModel;

    //OpenNLP tools are not thread safe, need ThreadLocals for each thread to have its own copy
    private ThreadLocal<SentenceDetector> ssplit;
    private ThreadLocal<POSTaggerME> postagger;
    private ThreadLocal<TokenizerME> tokenizer;

    @Override
    public List<CoreLabel> tokens(String text) {
        return toCoreLabel(tokenizer.get().tokenize(text), null);
    }

    @Override
    public List<CoreMap> sentences(String text) {
        String[] sentences = ssplit.get().sentDetect(text);
        ArrayList<CoreMap> sents = new ArrayList<>(sentences.length);
        for (String sentence : sentences) {
            List<CoreLabel> tokens = tokens(sentence);
            ArrayCoreMap map = new ArrayCoreMap();
            map.set(TokensAnnotation.class, tokens);
            sents.add(map);
        }
        return sents;
    }

    private OpenNLP() {
        //Singleton
        try {
            //Init OpenNLP
            tokenizerModel = new TokenizerModel(TagArticlesFromDatabase.class.getResourceAsStream("/en-token.bin"));
            sentenceModel = new SentenceModel(TagArticlesFromDatabase.class.getResourceAsStream("/en-sent.bin"));
            posModel = new POSModel(TagArticlesFromDatabase.class.getResourceAsStream("/en-pos-maxent.bin"));

            tokenizer = new ThreadLocal<TokenizerME>() {
                @Override
                protected TokenizerME initialValue() {
                    return new TokenizerME(tokenizerModel);
                }
            };
            ssplit = new ThreadLocal<SentenceDetector>() {
                @Override
                protected SentenceDetector initialValue() {
                    return new SentenceDetectorME(sentenceModel);
                }
            };
            postagger = new ThreadLocal<POSTaggerME>() {
                @Override
                protected POSTaggerME initialValue() {
                    return new POSTaggerME(posModel);
                }
            };
        } catch (IOException ex) {
            Logger.getLogger(OpenNLP.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    private static OpenNLP INSTANCE;
    public static OpenNLP get() {
        if (INSTANCE == null) {
            INSTANCE = new OpenNLP();
        }
        return INSTANCE;
    }

}
