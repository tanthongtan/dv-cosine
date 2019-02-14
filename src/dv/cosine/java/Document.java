package dv.cosine.java;

public class Document {
    
    public int[] wordIds;
    public int tag;
    public String split;
    public int sentiment;
    
    public Document(int[] wordIds, int tag, String split, int sentiment) {
        this.wordIds = wordIds;
        this.tag = tag;
        this.split = split;
        this.sentiment = sentiment;
    }
    
}
