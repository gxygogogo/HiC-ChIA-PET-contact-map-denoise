import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()

# Nodes
G.add_node("x1", label="Input x1")
G.add_node("x2", label="Input x2")
G.add_node("x3", label="Input x3")
G.add_node("SM Encoder", label="SM Encoder")
G.add_node("SA1 Encoder", label="SA1 Encoder")
G.add_node("SA2 Encoder", label="SA2 Encoder")
G.add_node("Dropout1", label="Dropout")
G.add_node("Dropout2", label="Dropout")
G.add_node("Dropout3", label="Dropout")
G.add_node("SM Decoder", label="SM Decoder")
G.add_node("SA1 Decoder", label="SA1 Decoder")
G.add_node("SA2 Decoder", label="SA2 Decoder")
G.add_node("Dropout4", label="Dropout")
G.add_node("Dropout5", label="Dropout")
G.add_node("Dropout6", label="Dropout")
G.add_node("Add1", label="Add(x1)")
G.add_node("Add2", label="Add(x2)")
G.add_node("Add3", label="Add(x3)")
G.add_node("ReLU1", label="ReLU")
G.add_node("ReLU2", label="ReLU")
G.add_node("ReLU3", label="ReLU")
G.add_node("x1_decoded", label="Output x1_decoded")
G.add_node("x2_decoded", label="Output x2_decoded")
G.add_node("x3_decoded", label="Output x3_decoded")
G.add_node("SM_encoded", label="SM_encoded")
G.add_node("SA1_encoded", label="SA1_encoded")
G.add_node("SA2_encoded", label="SA2_encoded")

# Edges
G.add_edges_from([
    ("x1", "SM Encoder"),
    ("x2", "SA1 Encoder"),
    ("x3", "SA2 Encoder"),
    ("SM Encoder", "Dropout1"),
    ("SA1 Encoder", "Dropout2"),
    ("SA2 Encoder", "Dropout3"),
    ("Dropout1", "SM_encoded"),
    ("Dropout2", "SA1_encoded"),
    ("Dropout3", "SA2_encoded"),
    ("Dropout1", "SM Decoder"),
    ("Dropout2", "SA1 Decoder"),
    ("Dropout3", "SA2 Decoder"),
    ("SM Decoder", "Dropout4"),
    ("SA1 Decoder", "Dropout5"),
    ("SA2 Decoder", "Dropout6"),
    ("Dropout4", "Add1"),
    ("Dropout5", "Add2"),
    ("Dropout6", "Add3"),
    ("x1", "Add1"),
    ("x2", "Add2"),
    ("x3", "Add3"),
    ("Add1", "ReLU1"),
    ("Add2", "ReLU2"),
    ("Add3", "ReLU3"),
    ("ReLU1", "x1_decoded"),
    ("ReLU2", "x2_decoded"),
    ("ReLU3", "x3_decoded")
])

pos = nx.spring_layout(G)

plt.figure(figsize=(14, 10))
nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
plt.title("Network Architecture Diagram")
plt.savefig('/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/Network_Architecture.pdf')
