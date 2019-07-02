package siddhesh.sparkl;

import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.tokensregex.MultiPatternMatcher;
import edu.stanford.nlp.ling.tokensregex.SequenceMatchResult;
import edu.stanford.nlp.ling.tokensregex.SequencePattern;
import edu.stanford.nlp.ling.tokensregex.TokenSequencePattern;
import edu.stanford.nlp.util.CoreMap;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.rdf.model.Literal;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Property;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.rdf.model.ResourceFactory;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.util.iterator.ExtendedIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.TaskContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.util.LongAccumulator;
import scala.collection.mutable.WrappedArray;

/**
 * Tags sentences with entities and relations in articles served by SPARQL server. This is an older approach in which
 * the article text is contained as a property in the triple store. One describe query can yield both articles as well
 * as all the relations it contains. The advantage is that you don't have to send the Wikipedia text dump to every
 * machine in the cluster, which can be slow when your network can't handle the load (case in point: COEP LAN). Every
 * machine simply sends a request to the server and computation begins as soon as you fire up.
 *
 * The downside is that your database server becomes the performance bottleneck. You can scale with more database
 * servers, but the latency incurred by each individual database also slows down execution. Secondly, the database size
 * unnecessarily inflates due to article text, which is processed only once. practical tests show that when the entire
 * database is not memory mapped in RAM, the query response becomes 1000x slower. With the smaller summary, DB size for
 * DBpedia dataset was 19GB , with full article text it inflated to 39GB. Without article text, DB size was 15GB.
 *
 * Always remember the time and efforts.
 *
 * <pre>
 * <code>
 * TDB database creation time
 * INFO Total: 199,597,131 tuples 2,941.31 seconds : 67,860.02 tuples/sec [2018/03/31 21:45:22 IST] 22:13:09
 * INFO Index Building Phase Completed 22:13:09
 * INFO -- TDB Bulk Loader Finish 22:13:09
 * INFO -- 4609 seconds
 * </code>
 * </pre>
 *
 * @author Siddhesh Rane
 */
public class TagArticlesFromDatabase implements Serializable {

    // Sparql and Jena Fuseki specific
    final static Property LABEL_PROPERTY = ResourceFactory.createProperty("http://www.w3.org/2000/01/rdf-schema#label");
    final static Property ABSTRACT_PROPERTY = ResourceFactory.createProperty("http://dbpedia.org/ontology/abstract");
    final static Property LINK_PROPERTY = ResourceFactory.createProperty("http://dbpedia.org/ontology/wikiPageWikiLink");
    final static Property SEEALSO_PROPERTY = ResourceFactory.createProperty("http://www.w3.org/2000/01/rdf-schema#seeAlso");
    final static String PREFIX = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                                 + "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                                 + "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                                 + "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
                                 + "PREFIX dbo: <http://dbpedia.org/ontology/>\n";

    public static final String FUSEKI_DBPEDIA_LOCAL = "http://localhost:3030/dbpedia/query";
    public static final Encoder<RelationSentence> RELATION_ENCODER = Encoders.bean(RelationSentence.class);

    /**
     * @param args the command line arguments
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws IOException {
        String[] sparqlServers = {FUSEKI_DBPEDIA_LOCAL};
        int range = args.length > 0 ? Integer.parseInt(args[0]) : 10;
        int batch = args.length > 1 ? Integer.parseInt(args[1]) : 50;
        if (args.length > 2) {
            sparqlServers = Stream.of(args).skip(2).map(ip -> "http://" + ip + "/dbpedia/query").toArray(String[]::new);
        }

        new TagArticlesFromDatabase(range, batch, sparqlServers).extractTaggedSentences();
//        new TagArticlesFromDatabase(range, batch, sparqlServers).testNullSubjects();
    }

    final String[] sparqlServers;
    // batch size: this controls the number of entities that we request in one call to sparql server
    final int batch;
    //number of batches processed
    final int range;

    SparkSession spark;
    LongAccumulator labelledCount;
    LongAccumulator unlabelledCount;
    LongAccumulator totalEntities;
    LongAccumulator skippedEntities;
    LongAccumulator numSentences;
    LongAccumulator numRelationalSentences;

    public TagArticlesFromDatabase(int range, int batchsize, final String... fusekiServer) {
        this.range = range;
        batch = batchsize;
        sparqlServers = fusekiServer;
        // Init Spark Context
        SparkConf conf = new SparkConf().setMaster("local");
        spark = SparkSession.builder()
            //.config(conf)
            .appName("Sid2Sparkl").getOrCreate();

        // create accumulators to track progress
        labelledCount = spark.sparkContext().longAccumulator("labelled");
        unlabelledCount = spark.sparkContext().longAccumulator("unlabelled");
        totalEntities = spark.sparkContext().longAccumulator("entities-traversed");
        skippedEntities = spark.sparkContext().longAccumulator("entities-skipped");
        numSentences = spark.sparkContext().longAccumulator("sentences");
        numRelationalSentences = spark.sparkContext().longAccumulator("sentencesWithRelation");
    }

    public final QueryExecution q(final String query) {
        //Use round robin load balancing based on current partition number
        int id = TaskContext.get().partitionId();
        String sparqlServer = sparqlServers[id % sparqlServers.length];
        return QueryExecutionFactory.sparqlService(sparqlServer, PREFIX + query);
    }

    public void extractTaggedSentences() {
        Dataset relations = spark
            .range(0, range, 1, range)
            .map(this::getSubjects, Encoders.bean(Model.class))
            .flatMap(this::extractRelationalSentences, RELATION_ENCODER);

        spark.udf().register("stringify", functions.udf((WrappedArray<String> arg0) -> arg0.mkString(" "), DataTypes.StringType));
        Dataset toCSV = relations.repartition(col("relation"))
            .withColumn("leftcsv", functions.callUDF("stringify", col("left")))
            .withColumn("rightcsv", functions.callUDF("stringify", col("right")))
            .withColumn("entity", functions.callUDF("stringify", col("entity")))
            .select("leftcsv", "entity", "rightcsv", "relation", "subject");

        final String path = "relation-" + range + "x" + batch + '-'
                            + (Boolean.getBoolean("nlp.opennlp") ? "opennlp" : "corenlp") + ".tsv";
        toCSV.write()
            .partitionBy("relation")
            .mode(SaveMode.Overwrite)
            .option("sep", "\t")
            .option("header", "true")
            .option("compression", "gzip")
            .csv(path);
        spark.stop();
    }

    public Model getSubject(String uri) {
        Model model = q("DESCRIBE " + uri).execDescribe();
        populateLabels(model);
        return model;
    }

    public Model getSubjects(long offset) {
        return getSubjects(offset * batch, batch);
    }

    public Model getSubjects(long offset, int limit) {
        // UPDATE: By adding dbo:wikiPageWikiLink triples we now have 200M triples which slowed down the
        // earlier combined query
        // this split query is more efficient by orders of magnitude
        String qdescribeEntities = "DESCRIBE ?s WHERE { ?s dbo:abstract [] .} OFFSET " + offset + " LIMIT " + limit;
        Model model = q(qdescribeEntities).execDescribe();
        totalEntities.add(limit);
        populateLabels(model);
        return model;
    }

    /**
     * Fetches labels for all the uri type objects in the model and updates the model. Note that uris in the subject
     * field are not considered. Generally, a model is first constructed using <code>DESCRIBE</code> query, and then
     * this method is called on the resulting model.
     *
     * @param model An existing model containing resource(URI) objects in triples.
     * @return the same model with added label triples
     */
    public Model populateLabels(Model model) {
        // get labels for all objects
        StringBuilder qlabels = new StringBuilder("CONSTRUCT { ?obj rdfs:label ?label } WHERE { VALUES ?obj { ");
        model.listObjects()
            .filterKeep(RDFNode::isURIResource)
            .mapWith(RDFNode::asResource)
            .mapWith(Resource::getURI)
            .forEachRemaining(t -> qlabels.append(" <").append(t).append(">"));
        qlabels.append("} \n  ?obj rdfs:label ?label . }");
        q(qlabels.toString()).execConstruct(model);

        // some objects dont have labels and are useless for our purpose. Remove them
        // UPDATE 5 April 2018: some subjects also dont have labels, even if they have abstracts. Remove
        // them too
        Set<Resource> labelled = model.listSubjectsWithProperty(LABEL_PROPERTY).toSet();
        Set<Resource> unlabelled = model.listObjects()
            .filterKeep(RDFNode::isResource)
            .andThen(model.listSubjects())
            .mapWith(RDFNode::asResource)
            .filterDrop(labelled::contains)
            .toSet();

        labelledCount.add(labelled.size());
        unlabelledCount.add(unlabelled.size());
        for (Resource resource : unlabelled) {
            model.removeAll(null, null, resource);
            model.removeAll(resource, null, null);
        }
        return model;
    }

    public Iterator<RelationSentence> extractRelationalSentences(Model model) {
        List<Resource> subjects = model.listSubjectsWithProperty(ABSTRACT_PROPERTY).toList();
        // prepare label cloud, mapping an RDFNode which could be a literal or a URI, to its string
        // representation
        HashMap<RDFNode, String> labelcloud = new HashMap<>();
        for (Resource person : subjects) {
            Statement abs = person.getProperty(ABSTRACT_PROPERTY);
            Statement personlabel = person.getProperty(LABEL_PROPERTY);
            person.listProperties().filterDrop(abs::equals).filterDrop(s -> s.equals(personlabel))
                .forEachRemaining(s -> {
                    RDFNode object = s.getObject();
                    Literal label = object.isLiteral()
                        ? object.asLiteral()
                        : object.asResource().getProperty(LABEL_PROPERTY).getLiteral();
                    labelcloud.putIfAbsent(object, label.getString().toLowerCase());
                    // System.out.println(s.getSubject().getLocalName() + "\t" +
                    // s.getPredicate().getLocalName() + "\t" + s.getObject().toString() + "\t" +
                    // labelcloud.get(entity));
                });
            labelcloud.putIfAbsent(person, personlabel.getString());
        }

        // create regex patterns for labels
        long time = System.currentTimeMillis();
        HashMap<TokenSequencePattern, RDFNode> pattern2node = new HashMap<>();
        for (Map.Entry<RDFNode, String> entry : labelcloud.entrySet()) {
            RDFNode obj = entry.getKey();
            String label = entry.getValue();
            List<CoreLabel> tokens = NLP.getTokens(label);
            Set<TokenSequencePattern> patterns = NLP.createPrefixSuffixPatterns(tokens);
            for (TokenSequencePattern pattern : patterns) {
                pattern2node.put(pattern, obj);
            }
        }
        MultiPatternMatcher<CoreMap> multimatcher = TokenSequencePattern.getMultiPatternMatcher(pattern2node.keySet());
        time = System.currentTimeMillis() - time;
        System.out.println("tok+pos+pattern generation = " + time);

        // now match in abstract
        ArrayList<RelationSentence> relationTaggedSentences = new ArrayList<>();
        for (Resource subject : subjects) {
            // List relations other than dbo:wikiPageWikiLink
            HashMap<RDFNode, Set<Property>> node2prop = new HashMap();
            Statement abs = subject.getProperty(ABSTRACT_PROPERTY);
            Statement label = subject.getProperty(LABEL_PROPERTY);
            Set<Statement> links = subject.listProperties(LINK_PROPERTY)
                .andThen(subject.listProperties(SEEALSO_PROPERTY)).toSet();
            ExtendedIterator<Statement> meaningfulRelations = subject.listProperties()
                .filterDrop(links::contains).filterDrop(abs::equals).filterDrop(label::equals);

            meaningfulRelations.forEachRemaining(s -> {
                Property relation = s.getPredicate();
                RDFNode object = s.getObject();
                Set<Property> props = node2prop.getOrDefault(object, new HashSet<>());
                props.add(relation);
                node2prop.put(object, props);
            });
            if (node2prop.isEmpty()) {
                // this entity does not have any useful properties, Skip it
                skippedEntities.add(1);
                continue;
            }

            List<CoreMap> sentences = NLP.getSentences(abs.getString());
            numSentences.add(sentences.size());
            for (CoreMap sentence : sentences) {
                boolean containsRelation = false;
                List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
                List<SequenceMatchResult<CoreMap>> matches = multimatcher.findNonOverlapping(tokens);
                for (SequenceMatchResult<CoreMap> match : matches) {
                    SequencePattern<CoreMap> pattern = match.pattern();
                    @SuppressWarnings("element-type-mismatch")
                    RDFNode node = pattern2node.get(pattern);
                    Set<Property> relation = node2prop.get(node);
                    if (relation == null) {
                        // This match does not give any useful relation
                        // ignore it
                        continue;
                    }
                    containsRelation = true;
                    List<CoreLabel> matched = (List<CoreLabel>) match.groupNodes();
                    int start = matched.get(0).index(); // 1 based indexing
                    int end = matched.get(matched.size() - 1).index();
                    String[] left = tokens.subList(0, start - 1).stream().map(CoreLabel::word).toArray(String[]::new);
                    String[] right = tokens.subList(end, tokens.size()).stream().map(CoreLabel::word)
                        .toArray(String[]::new);
                    String[] entity = matched.stream().map(CoreLabel::word).toArray(String[]::new);
                    for (Property p : relation) {
                        String rel = p.getLocalName();
                        rel = rel.isEmpty() ? p.getURI() : rel;
                        RelationSentence rs = new RelationSentence(left, entity, right, rel, subject.getURI());
                        relationTaggedSentences.add(rs);
                    }
                }
                if (containsRelation) {
                    numRelationalSentences.add(1);
                }
            }
        }
        return relationTaggedSentences.iterator();
    }

    public void testNullSubjects() {
        spark.range(1).foreach(l
            -> {
            Model m = q(PREFIX + "describe * where { ?p dbo:birthPlace <http://dbpedia.org/resource/Armagh> .}").execDescribe();
            System.out.println(m.getNsPrefixMap());
            m.listSubjects().forEachRemaining(s -> {
                String localName = s.getLocalName();
                System.out.println("" + s.getURI() + "\t\t[" + s.getNameSpace() + "|" + localName + "]");
            });
        }
        );
    }

    /**
     * Bean class to capture a relation containing sentence split into {@code left} {@code entity} {@code right}. All
     * getters, setters and argument constructors are created automatically by lombok annotations during build.
     *
     */
    @Data
    @AllArgsConstructor
    public static class RelationSentence implements Serializable {

        private String left[];
        private String entity[];
        private String right[];
        private String relation;
        private String subject;

        public RelationSentence() {
        }

    }

}
