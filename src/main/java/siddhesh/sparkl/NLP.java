package siddhesh.sparkl;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.tokensregex.Env;
import edu.stanford.nlp.ling.tokensregex.NodePattern;
import edu.stanford.nlp.ling.tokensregex.TokenSequencePattern;
import edu.stanford.nlp.util.CoreMap;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;
import java.util.stream.Collector;
import java.util.stream.Collectors;

/**
 * This class abstracts NLP tokenization and sentence splitting, using the
 * CoreNLP API, with configurable backend using either OpenNLP or CoreNLP.
 *
 * Uses Stanford CoreNLP by default. to use OpenNLP, supply the system property
 * {@code -Dnlp.opennlp=true} on the command line.
 *
 * @author Siddhesh Rane
 */
public abstract class NLP {

    static final NLP impl;
    static {
        if (Boolean.getBoolean("nlp.opennlp")) {
            impl = OpenNLP.get();
            System.out.println("Using OpenNLP");
        } else {
            impl = CoreNLP.get();
            System.out.println("Using CoreNLP");
        }

    }

    /**
     * Converts an array of string tokens and pos tags to CoreNLP CoreLabels.
     * This method fill in {@code word}s, {@code tag}s and {@code index} fields
     * in {@code CoreLabel}. tags can be null, or less than/more than words.
     * Extra or missing tags will be null. Indexing starts from 1.
     *
     * @param tokens list of tokens
     * @param tags   part of speech tags corresponding to tokens. Can be null.
     * @return a list of corelabels encapsulating the above information
     */
    public static List<CoreLabel> toCoreLabel(String[] tokens, String[] tags) {
        ArrayList<CoreLabel> coreLabels = new ArrayList<>(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];
            CoreLabel cl = new CoreLabel();
            token = token.replace("\"", "''");
            cl.setWord(token);
            cl.setOriginalText(token);
            cl.setIndex(i + 1);
            if (tags != null && tags.length > i) {
                cl.setTag(tags[i]);
            }
            coreLabels.add(cl);
        }
        return coreLabels;
    }
    /**
     * This method will create a group of TokensRegex Patterns for the given
     * token sequence, which will match the suffix, prefix or some combination
     * that is usually used in natural language for repeated mentions. In this
     * implementation we handle 3 cases common to Wikipedia title names.
     *
     * <ol>
     * <li>The University of Minnesota -> university, Minnesota. </li>
     * <li>New York University -> new, new york, york university,
     * university</li>
     * <li>Western philosophy -> western philosophy, philosophy, western .*
     * philosophy</li>
     * </ol>
     *
     * @param tokens
     * @return Regex patterns for most likely prefixes/suffixes
     */
    public static Set<TokenSequencePattern> createPrefixSuffixPatterns(List<CoreLabel> tokens) {
        //filter the tokens list to remove (words in parens). and comma preceding elements as they are just superfluous details that mislead NER
        ArrayList<CoreLabel> filtered = new ArrayList<>(tokens.size());
        boolean inParens = false;
        for (CoreLabel token : tokens) {
            String word = token.word();
            if ("-LRB-".equals(word) || "(".equals(word)) {
                inParens = true;
                continue;
            } else if ("-RRB-".equals(word) || ")".equals(word)) {
                inParens = false;
                continue;
            }
            if (inParens) {
                continue;
            }
            if (",".equals(word)) {
                break;
            }
            filtered.add(token);
        }
        HashSet<TokenSequencePattern> patterns = new HashSet<>();
        Env env = TokenSequencePattern.getNewEnv();
        env.setDefaultStringMatchFlags(NodePattern.CASE_INSENSITIVE);
        env.setDefaultStringPatternFlags(Pattern.CASE_INSENSITIVE);
        final Collector<CharSequence, ?, String> tokenListToRegex = Collectors.joining("\"}] [{word:\"", "[{word:\"", "\"}]");
        //        Collector<CharSequence, ?, String> joinbyregex = Collectors.joining("/ /", "/", "/");
        Collector<CharSequence, ?, String> joinbyregex = Collectors.joining("\" \"", "\"", "\"");
        //        Function<CoreLabel,String> toRegexToken = (CoreLabel l)->Pattern.quote(l.word().replace("/", "\\/"));
        Function<CoreLabel, String> toRegexToken = CoreLabel::word;
        final int size = filtered.size();
        final String pattern = filtered.stream().map(toRegexToken).collect(joinbyregex);
        try {
            TokenSequencePattern fullMatch = TokenSequencePattern.compile(env, pattern);
//            fullMatch.setPriority(1);
            patterns.add(fullMatch);
        } catch (Exception ex) {
            Logger.getLogger(NLP.class.getName()).log(Level.WARNING, ex.getMessage());
        }
        //
        //        Predicate<CoreLabel> validStart = t -> {
        //            char c = t.tag().charAt(0);
        //            return c == 'N' || c == 'V' || c == 'J' || c == 'R' || "CD".equals(t.tag());
        //        };
        //        Predicate<CoreLabel> validEnd = t -> {
        //            char c = t.tag().charAt(0);
        //            return c == 'N' || c == 'V' || "CD".equals(t.tag());
        //        };
        //
        //        for (int i = 1, j = size - 1; i < size; i++, j--) {
        //            if (validStart.test(filtered.get(i)) && (i == size - 1 && !"CD".equals(filtered.get(i).tag()))) {
        //                //for suffix
        //                final String regex = filtered.subList(i, size).stream().map(toRegexToken).collect(joinbyregex);
        //                TokenSequencePattern pattern = TokenSequencePattern.compile(env, regex);
        //                pattern.setPriority(j / size);
        //                patterns.add(pattern);
        //            }
        //            if (validEnd.test(filtered.get(j - 1)) && (i == size - 1 && !"CD".equals(filtered.get(j - 1).tag()))) {
        //                //for prefix
        //                final String regex = filtered.subList(0, j).stream().map(toRegexToken).collect(joinbyregex);
        //                TokenSequencePattern pattern = TokenSequencePattern.compile(env, regex);
        //                pattern.setPriority(i / size);
        //                patterns.add(pattern);
        //            }
        //        }
        return patterns;
    }

    protected NLP() {
        //Singleton
    }

    public static List<CoreLabel> getTokens(String text) {
        return impl.tokens(text);
    }
    public static List<CoreMap> getSentences(String text) {
        return impl.sentences(text);
    }

    abstract List<CoreLabel> tokens(String text);
    abstract List<CoreMap> sentences(String text);

}
