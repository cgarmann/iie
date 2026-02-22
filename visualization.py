import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

SAMPLE_SIZE = 2000
VIZ_METHOD = 'pca'
OUTPUT_PREFIX = f'semantic_map_{VIZ_METHOD}_'

CATEGORY_COLORS = {
    'AI & Robotics': '#FF2F92',
    'Cybersecurity': '#00B3A4',
    'Health': '#008CBE',
    'Energy & Environment': '#3A9C3A',
    'Space': '#FFC300',
    'Ocean': '#7B1FA2',
    'Education': '#00695C',
    'Food & Biotech': '#F39C12',
}

df = pd.read_csv("idea_sample.csv")

if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = []
for text in tqdm(df['OriginalText'], desc="Encoding"):
    embeddings.append(model.encode(text))

embeddings = np.array(embeddings)

if VIZ_METHOD == 'pca':
    reducer = PCA(n_components=2, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)
else:
    reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    coords_2d = reducer.fit_transform(embeddings)

df['x'] = coords_2d[:, 0]
df['y'] = coords_2d[:, 1]

plt.figure(figsize=(16, 10), facecolor='#0E1117')
ax = plt.gca()
ax.set_facecolor('#0E1117')

for category in df['Category'].unique():
    mask = df['Category'] == category
    color = CATEGORY_COLORS.get(category, '#95A5A6')
    
    plt.scatter(
        df[mask]['x'], 
        df[mask]['y'],
        c=color,
        label=category,
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )

plt.title('Semantic Landscape of Ideas', fontsize=24, color='white', pad=20)
plt.xlabel(f'{VIZ_METHOD.upper()} Component 1', fontsize=14, color='white')
plt.ylabel(f'{VIZ_METHOD.upper()} Component 2', fontsize=14, color='white')
plt.legend(loc='upper right', framealpha=0.9, fontsize=10, facecolor='#1E1E1E', edgecolor='white')
plt.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

static_filename = f'static_{OUTPUT_PREFIX}{len(df)}.png'
plt.savefig(static_filename, dpi=300, bbox_inches='tight', facecolor='#0E1117')
plt.close()

fig = go.Figure()

for category in sorted(df['Category'].unique()):
    mask = df['Category'] == category
    category_df = df[mask]
    
    color = CATEGORY_COLORS.get(category, '#95A5A6')
    
    fig.add_trace(go.Scatter(
        x=category_df['x'],
        y=category_df['y'],
        mode='markers',
        name=category,
        marker=dict(
            size=8,
            color=color,
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=category_df['OriginalText'],
        hovertemplate='<b>%{text}</b><br>Category: ' + category + '<extra></extra>',
        showlegend=True
    ))

fig.update_layout(
    title={
        'text': 'Interactive Semantic Landscape',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 28, 'color': 'white'}
    },
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font=dict(color='white', size=12),
    xaxis=dict(
        title=f'{VIZ_METHOD.upper()} Component 1',
        showgrid=True,
        gridcolor='#333333',
        zeroline=False
    ),
    yaxis=dict(
        title=f'{VIZ_METHOD.upper()} Component 2',
        showgrid=True,
        gridcolor='#333333',
        zeroline=False
    ),
    height=700,
    hovermode='closest',
    legend=dict(
        bgcolor='rgba(30,30,30,0.8)',
        bordercolor='white',
        borderwidth=1,
        font=dict(size=11)
    )
)

interactive_filename = f'interactive_{OUTPUT_PREFIX}{len(df)}.html'
fig.write_html(interactive_filename)

print("Done")
