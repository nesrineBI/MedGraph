import spacy
import scispacy
from scispacy.linking import EntityLinker
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt


# Load the domain-specific medical model
nlp = spacy.load("en_core_sci_sm")

# Load the NER model for biomedical entities
nlp_ner = spacy.load("en_ner_bc5cdr_md")

# Read the medical document
with open('medical_note.txt', 'r') as file:
    medical_note = file.read()

print(medical_note[:500])  # Display an excerpt of the document for verification

# Process the medical text with SpaCy
doc = nlp(medical_note)

# Extract named entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Entities:", entities)

# Process the medical text with the biomedical NER model
doc_ner = nlp_ner(medical_note)

# Extract biomedical named entities
entities_biomedical = [(ent.text, ent.label_) for ent in doc_ner.ents]
print("Biomedical Entities:", entities_biomedical)

# Extract relationships between entities (simple proximity-based relations)
relations = []
for ent in doc_ner.ents:
    for ent2 in doc_ner.ents:
        if ent != ent2:
            relations.append((ent.text, ent2.text))

print("Relations:", relations)

# Initialize an undirected graph
G = nx.Graph()

# Add entities as nodes
for entity, label in entities_biomedical:
    G.add_node(entity, label=label)

# Add relations as edges
for ent1, ent2 in relations:
    G.add_edge(ent1, ent2)

# Draw the graph
pos = nx.spring_layout(G)  # for visually pleasing node layout
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
plt.title("Knowledge Graph of Medical Notes")
plt.show()

# Visualize recognized entities in the medical document
displacy.render(doc_ner, style="ent", jupyter=True)


