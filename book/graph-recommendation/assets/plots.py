import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
OUT="/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/graph-recommendation/assets/"
plt.rcParams.update({'figure.dpi':130,'font.size':11,'axes.grid':True,'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False,'figure.autolayout':True})
BLUE,ORANGE,GREEN,RED,GRAY='#2563eb','#ea7317','#16a34a','#dc2626','#64748b'

# 1) link-prediction AUC by method family (illustrative)
fig,ax=plt.subplots(figsize=(7,3.8))
methods=['Common\nNeighbors','Adamic-\nAdar','Node2Vec','GraphSAGE\n(inductive GNN)','Subgraph GNN\n(SEAL-style)']
auc=[0.82,0.86,0.90,0.93,0.95]
colors=[GRAY,GRAY,ORANGE,BLUE,GREEN]
b=ax.bar(methods,auc,color=colors,alpha=0.85)
ax.set_ylim(0.75,1.0); ax.set_ylabel('link-prediction AUC')
ax.set_title('Link prediction: heuristics are strong baselines,\nGNNs win when features and structure both matter (illustrative)')
ax.axhline(0.86,ls='--',color=GRAY,lw=1); ax.text(0.0,0.865,'a good heuristic is hard to beat cheaply',fontsize=8,color=GRAY)
fig.savefig(OUT+"fig-linkpred-auc.png"); plt.close(fig)

# 2) GraphSAGE 2-hop neighbor sampling schematic
fig,ax=plt.subplots(figsize=(6,4)); ax.axis('off')
ax.set_title('Inductive GNN: sample a fixed fan-out of neighbors,\naggregate inward to embed the target node')
import math
# target
ax.scatter([0],[0],s=900,color=BLUE,zorder=5); ax.text(0,0,'v',color='white',ha='center',va='center',fontsize=13,zorder=6)
# 1-hop (3 nodes)
h1=[(-1.4,1.1),(0,1.5),(1.4,1.1)]
for x,y in h1:
    ax.plot([0,x],[0,y],color=GRAY,lw=1.5,zorder=1); ax.scatter([x],[y],s=520,color=ORANGE,zorder=5)
# 2-hop (2 each)
for (x,y) in h1:
    for dx in (-0.7,0.7):
        x2,y2=x+dx,y+1.0
        ax.plot([x,x2],[y,y2],color=GRAY,lw=1,alpha=0.6,zorder=1); ax.scatter([x2],[y2],s=240,color=GREEN,alpha=0.8,zorder=4)
ax.scatter([],[],color=BLUE,label='target node v'); ax.scatter([],[],color=ORANGE,label='1-hop sample'); ax.scatter([],[],color=GREEN,label='2-hop sample')
ax.legend(loc='lower center',ncol=3,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.08))
ax.set_xlim(-3,3); ax.set_ylim(-0.6,3)
fig.savefig(OUT+"fig-neighbor-sampling.png"); plt.close(fig)

# 3) degree distribution (power law) motivating negative sampling correction
fig,ax=plt.subplots(figsize=(6,3.6))
rank=np.arange(1,5001); deg=6000.0/rank**0.85
ax.loglog(rank,deg,color=BLUE,lw=2)
ax.set_xlabel('node rank by degree (log)'); ax.set_ylabel('degree (log)')
ax.set_title('Social graphs are power-law: a few hub nodes dominate,\nso uniform negatives over-sample hubs (correct for it)')
ax.annotate('hubs (celebrities,\nbig pages)',xy=(2,deg[1]),xytext=(20,deg[1]*0.6),fontsize=9,color=GRAY,arrowprops=dict(arrowstyle='->',color=GRAY))
fig.savefig(OUT+"fig-degree-distribution.png"); plt.close(fig)
print("wrote 3 figures")
