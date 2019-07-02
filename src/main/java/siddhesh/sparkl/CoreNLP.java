package siddhesh.sparkl;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;
import java.util.List;

public class CoreNLP extends NLP {

    private static CoreNLP INSTANCE;
    public static NLP get() {
        if (INSTANCE == null) {
            INSTANCE = new CoreNLP();
        }
        return INSTANCE;
    }

    private CoreNLP() {
        corenlp = new StanfordCoreNLP(PropertiesUtils.asProperties("annotators", "tokenize,ssplit"));
    }

    private final StanfordCoreNLP corenlp;

    @Override
    public List<CoreMap> sentences(String text) {
        Annotation annotation = new Annotation(text);
        corenlp.annotate(annotation);
        return annotation.get(SentencesAnnotation.class);
    }

    @Override
    public List<CoreLabel> tokens(String text) {
        Annotation annotation = new Annotation(text);
        corenlp.annotate(annotation);
        return annotation.get(TokensAnnotation.class);
    }

}
