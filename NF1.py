#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip -q install palantir fa2')


# In[2]:


import warnings
warnings.filterwarnings("ignore")
from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
import scFates as scf
import palantir
import matplotlib.pyplot as plt
sc.settings.verbosity = 3
sc.settings.logfile = sys.stdout
## fix palantir breaking down some plots
import seaborn
seaborn.reset_orig()
get_ipython().run_line_magic('matplotlib', 'inline')

sc.set_figure_params()
scf.set_figure_pubready()


# In[3]:


NF1 = sc.read_h5ad("/mnt/g/NF1.h5ad")


# In[4]:


NF1


# In[5]:


NF1.var


# In[6]:


pca_projections = pd.DataFrame(NF1.obsm["X_pca"],index=NF1.obs_names)


# In[7]:


dm_res = palantir.utils.run_diffusion_maps(pca_projections)
ms_data = palantir.utils.determine_multiscale_space(dm_res,n_eigs=4)


# In[8]:


# generate neighbor draph in multiscale diffusion space
NF1.obsm["X_palantir"]=ms_data.values
sc.pp.neighbors(NF1,n_neighbors=30,use_rep="X_palantir")


# In[9]:


NF1.obsm["X_pca2d"]=NF1.obsm["X_pca"][:,:2]
sc.tl.draw_graph(NF1,init_pos='X_pca2d') 


# In[10]:


sc.pl.draw_graph(NF1,color=["my_annotation_2",'group_id']) #PCA-based, n_neighbors = 30, n_eigs = 4


# In[11]:


sc.pl.umap(NF1,color=["my_annotation_2",'group_id']) #PCA-based, n_neighbors = 30, n_eigs = 4


# In[12]:


sc.pl.umap(NF1, color=["my_annotation_2"])


# In[13]:


sc.tl.leiden(NF1)


# In[14]:


sc.set_figure_params(figsize = [12,12])
sc.pl.umap(NF1, color=["seurat_clusters"], legend_loc = 'on data')


# In[ ]:





# In[15]:


NF1_test = NF1


# In[16]:


sc.pp.neighbors(NF1_test,n_neighbors=50,use_rep="X_palantir")


# In[17]:


NF1_test.obsm["X_pca2d"]=NF1_test.obsm["X_pca"][:,:2]
sc.tl.draw_graph(NF1_test,init_pos='X_pca2d') 


# In[18]:


sc.pl.draw_graph(NF1_test,color=["my_annotation_2",'group_id']) #PCA-based, n_neighbors = 30, n_eigs = 4


# In[45]:


NF1_Ctrl_WT = NF1[NF1.obs['group_id'].isin(['Ctrl1_WT'])]
NF1_Ctrl_NFflox = NF1[NF1.obs['group_id'].isin(['Ctrl2_NF1floxed'])]
NF1_Ctrl_NFhetero = NF1[NF1.obs['group_id'].isin(['Ctrl3_NF1hetero'])]
NF1_OPG_NFhomo = NF1[NF1.obs['group_id'].isin(['OPG_NF1homo'])]


# In[20]:


NF1_obs = pd.read_csv(r"//mnt//g//NF1_metadata.csv")


# In[21]:


NF1_obs


# In[22]:


NF1_obs = NF1_obs.rename(columns = {'Unnamed: 0':'CellID'})


# In[23]:


NF1_obs = NF1_obs.set_index('CellID')


# In[24]:


NF1_obs 


# In[25]:


NF1.obs['labels_SCT'] = NF1_obs['my_annotation_2']


# In[26]:


NF1.obs


# In[27]:


sc.pl.umap(NF1,color=["labels_SCT",'group_id'], legend_loc = 'on data')


# In[28]:


sc.pl.draw_graph(NF1,color=["labels_SCT",'group_id'], legend_loc = 'on data') #PCA-based, n_neighbors = 30, n_eigs = 4


# In[29]:


astrocytes = NF1[NF1.obs['labels_SCT'].isin(['Astro_1','Astro_2','Astro_3','Astro_4'])]


# In[30]:


pca_projections = pd.DataFrame(astrocytes.obsm["X_pca"],index=astrocytes.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections)
ms_data = palantir.utils.determine_multiscale_space(dm_res,n_eigs=4)
astrocytes.obsm["X_palantir"]=ms_data.values
sc.pp.neighbors(astrocytes,n_neighbors=30,use_rep="X_palantir")


# In[31]:


astrocytes.obsm["X_pca2d"]=astrocytes.obsm["X_pca"][:,:2]
sc.tl.draw_graph(astrocytes,init_pos='X_pca2d')


# In[32]:


sc.pl.draw_graph(astrocytes,color=["labels_SCT",'group_id'], legend_loc = 'on data')


# In[33]:


sc.pl.umap(astrocytes,color=["labels_SCT",'group_id'], legend_loc = 'on data')


# In[34]:


sc.tl.umap(astrocytes)


# In[35]:


sc.pl.umap(astrocytes,color=["labels_SCT",'group_id'], legend_loc = 'on data')


# In[36]:


def cluster_small_multiples(
    adata, clust_key, size=60, frameon=False, legend_loc=None, **kwargs
):
    tmp = adata.copy()

    for i, clust in enumerate(adata.obs[clust_key].cat.categories):
        tmp.obs[clust] = adata.obs[clust_key].isin([clust]).astype("category")
        tmp.uns[clust + "_colors"] = ["#d3d3d3", adata.uns[clust_key + "_colors"][i]]

    sc.pl.draw_graph(
        tmp,
        groups=tmp.obs[clust].cat.categories[1:].values,
        color=adata.obs[clust_key].cat.categories.tolist(),
        size=size,
        frameon=frameon,
        legend_loc=legend_loc,
        **kwargs,
    )


# In[37]:


def cluster_small_multiples_umap(
    adata, clust_key, size=60, frameon=False, legend_loc=None, **kwargs
):
    tmp = adata.copy()

    for i, clust in enumerate(adata.obs[clust_key].cat.categories):
        tmp.obs[clust] = adata.obs[clust_key].isin([clust]).astype("category")
        tmp.uns[clust + "_colors"] = ["#d3d3d3", adata.uns[clust_key + "_colors"][i]]

    sc.pl.umap(
        tmp,
        groups=tmp.obs[clust].cat.categories[1:].values,
        color=adata.obs[clust_key].cat.categories.tolist(),
        size=size,
        frameon=frameon,
        legend_loc=legend_loc,
        **kwargs,
    )


# In[38]:


cluster_small_multiples(NF1, clust_key = 'group_id', save = 'NF1.pdf')


# In[39]:


sc.pl.draw_graph(NF1,color=["labels_SCT"], legend_loc = 'right margin')


# In[40]:


NF1_Ctrl_WT 
NF1_Ctrl_NFflox 
NF1_Ctrl_NFhetero
NF1_OPG_NFhomo 


# In[46]:


sc.pl.draw_graph(NF1_Ctrl_WT,color=["labels_SCT"], legend_loc = 'on data')


# In[44]:


NF1_Ctrl_WT.obs


# In[ ]:


sc.pl.draw_graph(NF1_Ctrl_NFflox,color=["labels_SCT"], legend_loc = 'on data')


# In[ ]:


sc.pl.draw_graph(NF1_Ctrl_NFhetero,color=["labels_SCT"], legend_loc = 'on data')


# In[ ]:


#NF1_OPG_NFhomo 
sc.pl.draw_graph(NF1_OPG_NFhomo,color=["labels_SCT"], legend_loc = 'on data')


# In[57]:


sc.pl.draw_graph(NF1_Ctrl_WT,color=["labels_SCT"], legend_loc = 'on data')


# In[58]:


AnnData.write_h5ad(NF1, "/mnt/c/Users/Emil/10X/NF1merged1.h5ad")


# In[3]:


NF1 = sc.read_h5ad("/mnt/c/Users/Emil/10X/NF1merged1.h5ad")


# In[73]:


from adjustText import adjust_text


# In[81]:


def gen_mpl_labels(
    adata, groupby, exclude=(), ax=None, adjust_kwargs=None, text_kwargs=None
):
    if adjust_kwargs is None:
        adjust_kwargs = {"text_from_points": False}
    if text_kwargs is None:
        text_kwargs = {}

    medians = {}

    for g, g_idx in adata.obs.groupby(groupby).groups.items():
        if g in exclude:
            continue
        medians[g] = np.median(adata[g_idx].obsm["X_draw_graph_fa"], axis=0)

    if ax is None:
        texts = [
            plt.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()
        ]
    else:
        texts = [ax.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]

    adjust_text(texts, **adjust_kwargs)
    
with plt.rc_context({"figure.figsize": (6, 6), "figure.dpi": 300, "figure.frameon": False}):
    ax = sc.pl.draw_graph(NF1,color='labels_SCT', show=False, legend_loc='None', frameon=True)
    gen_mpl_labels(
        NF1,
        "labels_SCT",
        exclude=("None",),  # This was before we had the `nan` behaviour
        ax=ax,
        adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
        text_kwargs=dict(fontsize=12),
    )
    fig = ax.get_figure()
    fig.tight_layout()
    
    plt.savefig('figures/fa_NF1_new.pdf')
    plt.show()


# In[82]:


def gen_mpl_labels(
    adata, groupby, exclude=(), ax=None, adjust_kwargs=None, text_kwargs=None
):
    if adjust_kwargs is None:
        adjust_kwargs = {"text_from_points": False}
    if text_kwargs is None:
        text_kwargs = {}

    medians = {}

    for g, g_idx in adata.obs.groupby(groupby).groups.items():
        if g in exclude:
            continue
        medians[g] = np.median(adata[g_idx].obsm["X_umap"], axis=0)

    if ax is None:
        texts = [
            plt.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()
        ]
    else:
        texts = [ax.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]

    adjust_text(texts, **adjust_kwargs)
    
with plt.rc_context({"figure.figsize": (6, 6), "figure.dpi": 300, "figure.frameon": False}):
    ax = sc.pl.umap(NF1,color='labels_SCT', show=False, legend_loc='None', frameon=True)
    gen_mpl_labels(
        NF1,
        "labels_SCT",
        exclude=("None",),  # This was before we had the `nan` behaviour
        ax=ax,
        adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
        text_kwargs=dict(fontsize=12),
    )
    fig = ax.get_figure()
    fig.tight_layout()
    
    plt.savefig('figures/umap_NF1_new.pdf')
    plt.show()


# In[47]:


NF1_pop = NF1[NF1.obs['labels_SCT'].isin(['Olig_1'])]


# In[48]:


sc.pl.draw_graph(NF1_pop,color=["labels_SCT",'group_id'], legend_loc = 'on data')


# In[49]:


pca_projections = pd.DataFrame(NF1_pop.obsm["X_pca"],index=NF1_pop.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections)
ms_data = palantir.utils.determine_multiscale_space(dm_res,n_eigs=4)
NF1_pop.obsm["X_palantir"]=ms_data.values
sc.pp.neighbors(NF1_pop,n_neighbors=30,use_rep="X_palantir")
NF1_pop.obsm["X_pca2d"]=NF1_pop.obsm["X_pca"][:,:2]
sc.tl.draw_graph(NF1_pop,init_pos='X_pca2d')


# In[50]:


sc.pl.draw_graph(NF1_pop,color=["labels_SCT",'group_id'], legend_loc = 'on data')


# In[52]:


sc.pl.draw_graph(NF1_Ctrl_WT,color=["labels_SCT",'group_id'], legend_loc = 'on data')


# In[51]:


NF1_Ctrl_WT = NF1_pop[NF1_pop.obs['group_id'].isin(['Ctrl1_WT'])]
NF1_Ctrl_NFflox = NF1_pop[NF1_pop.obs['group_id'].isin(['Ctrl2_NF1floxed'])]
NF1_Ctrl_NFhetero = NF1_pop[NF1_pop.obs['group_id'].isin(['Ctrl3_NF1hetero'])]
NF1_OPG_NFhomo = NF1_pop[NF1_pop.obs['group_id'].isin(['OPG_NF1homo'])]


# In[53]:


import scvelo as scv


# In[54]:


WT1 = scv.read("/mnt/g/Aboozar_loom/7.loom", cache = True)
WT2 = scv.read("/mnt/g/Aboozar_loom/8.loom", cache = True)


# In[86]:


OPG1 = scv.read("/mnt/g/Aboozar_loom/1.loom", cache = True)
OPG2 = scv.read("/mnt/g/Aboozar_loom/2.loom", cache = True)
OPG3 = scv.read("/mnt/g/Aboozar_loom/3.loom", cache = True)
OPG4 = scv.read("/mnt/g/Aboozar_loom/4.loom", cache = True)


# In[152]:


NF1_Ctrl_WT_velo1 = scv.utils.merge(NF1_Ctrl_WT, WT1)


# In[153]:


NF1_OPG_NFhomo1 = scv.utils.merge(NF1_OPG_NFhomo, OPG1)
NF1_OPG_NFhomo2 = scv.utils.merge(NF1_OPG_NFhomo, OPG2)
NF1_OPG_NFhomo3 = scv.utils.merge(NF1_OPG_NFhomo, OPG3)
NF1_OPG_NFhomo4 = scv.utils.merge(NF1_OPG_NFhomo, OPG4)


# In[154]:


NF1_OPG_NFhomo_velo = NF1_OPG_NFhomo1.concatenate([NF1_OPG_NFhomo2, NF1_OPG_NFhomo3, NF1_OPG_NFhomo4])


# In[155]:


NF1_Ctrl_WT_velo2 = scv.utils.merge(NF1_Ctrl_WT, WT2)


# In[147]:


NF1_Ctrl_WT_velo1


# In[148]:


NF1_Ctrl_WT_velo2


# In[156]:


NF1_Ctrl_WT_velo = NF1_Ctrl_WT_velo1.concatenate([NF1_Ctrl_WT_velo2])


# In[150]:


NF1_total = NF1_Ctrl_WT_velo.concatenate([NF1_OPG_NFhomo_velo])


# In[64]:


NF1_Ctrl_WT_velo


# In[65]:


NF1_Ctrl_WT


# In[123]:


scv.pl.proportions(NF1_Ctrl_WT_velo)


# In[124]:


scv.pl.proportions(NF1_OPG_NFhomo_velo)


# In[125]:


scv.pp.filter_and_normalize(NF1_Ctrl_WT_velo, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(NF1_Ctrl_WT_velo, n_pcs=30, n_neighbors=30)


# In[126]:


scv.pp.filter_and_normalize(NF1_OPG_NFhomo_velo, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(NF1_OPG_NFhomo_velo, n_pcs=30, n_neighbors=30)


# In[127]:


scv.tl.velocity(NF1_OPG_NFhomo_velo)


# In[128]:


scv.tl.velocity(NF1_Ctrl_WT_velo)


# In[129]:


scv.tl.velocity_graph(NF1_Ctrl_WT_velo)


# In[130]:


scv.tl.velocity_graph(NF1_OPG_NFhomo_velo)


# In[71]:


NF1_Ctrl_WT_velo.obs


# In[74]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_Ctrl_WT_velo, color = 'labels_SCT', layout = 'fa')


# In[93]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_OPG_NFhomo_velo, color = 'labels_SCT', layout = 'fa')


# In[75]:


scv.pl.velocity_embedding_grid(NF1_Ctrl_WT_velo, basis='draw_graph_fa', color = 'labels_SCT', dpi = 300, add_margin = 0.2, figsize = (4,4), arrow_length = 4, arrow_size = 2, density = 1)


# In[131]:


scv.pl.velocity_embedding_grid(NF1_Ctrl_WT_velo, basis='draw_graph_fa', color = 'labels_SCT', dpi = 300, add_margin = 0.2, figsize = (4,4), arrow_length = 4, arrow_size = 2, density = 1)


# In[94]:


scv.pl.velocity_embedding_grid(NF1_OPG_NFhomo_velo, basis='draw_graph_fa', color = 'labels_SCT', dpi = 300, add_margin = 0.2, figsize = (4,4), arrow_length = 4, arrow_size = 2, density = 1)


# In[76]:


scv.pl.velocity_embedding_stream(NF1_Ctrl_WT_velo, basis='draw_graph_fa', color = 'labels_SCT', dpi = 300, density = 2, add_margin=0.2, arrow_size = 2, legend_loc = 'right margin')


# In[77]:


scv.tl.velocity_confidence(NF1_Ctrl_WT_velo)
keys =  'velocity_confidence','velocity_length'
scv.pl.scatter(NF1_Ctrl_WT_velo, c=keys, cmap='coolwarm', perc=[5, 95], basis='draw_graph_fa' )


# In[78]:


scv.tl.velocity_pseudotime(NF1_Ctrl_WT_velo)
scv.pl.scatter(NF1_Ctrl_WT_velo, color='velocity_pseudotime', cmap='gnuplot', basis='draw_graph_fa')


# In[95]:


scv.tl.velocity_pseudotime(NF1_OPG_NFhomo_velo)
scv.pl.scatter(NF1_OPG_NFhomo_velo, color='velocity_pseudotime', cmap='gnuplot', basis='draw_graph_fa')


# In[79]:


scv.tl.recover_dynamics(NF1_Ctrl_WT_velo)


# In[187]:


scv.tl.recover_dynamics(NF1_Ctrl_WT_velo2)


# In[96]:


scv.tl.recover_dynamics(NF1_OPG_NFhomo_velo)


# In[188]:


scv.tl.recover_dynamics(NF1_OPG_NFhomo_velo2)


# In[80]:


scv.tl.velocity(NF1_Ctrl_WT_velo, mode='dynamical')
scv.tl.velocity_graph(NF1_Ctrl_WT_velo)


# In[97]:


scv.tl.velocity(NF1_OPG_NFhomo_velo, mode='dynamical')
scv.tl.velocity_graph(NF1_OPG_NFhomo_velo)


# In[190]:


scv.tl.velocity(NF1_Ctrl_WT_velo2, mode='dynamical')
scv.tl.velocity_graph(NF1_Ctrl_WT_velo2)


# In[191]:


scv.tl.velocity(NF1_OPG_NFhomo_velo2, mode='dynamical')
scv.tl.velocity_graph(NF1_OPG_NFhomo_velo2)


# In[82]:


df = NF1_Ctrl_WT_velo.var
df = df[(df['fit_likelihood'] > .1) & df['velocity_genes'] == True]

kwargs = dict(xscale='log', fontsize=16)
with scv.GridSpec(ncols=3) as pl:
    pl.hist(df['fit_alpha'], xlabel='transcription rate', **kwargs)
    pl.hist(df['fit_beta'] * df['fit_scaling'], xlabel='splicing rate', xticks=[.1, .4, 1], **kwargs)
    pl.hist(df['fit_gamma'], xlabel='degradation rate', xticks=[.1, .4, 1], **kwargs)

scv.get_df(NF1_Ctrl_WT_velo, 'fit*', dropna=True).head()


# In[83]:


scv.tl.latent_time(NF1_Ctrl_WT_velo)
scv.pl.scatter(NF1_Ctrl_WT_velo, color='latent_time', color_map='gnuplot', size=80, basis = 'draw_graph_fa')


# In[98]:


scv.tl.latent_time(NF1_OPG_NFhomo_velo)
scv.pl.scatter(NF1_OPG_NFhomo_velo, color='latent_time', color_map='gnuplot', size=80, basis = 'draw_graph_fa')


# In[85]:


top_genes = NF1_Ctrl_WT_velo.var['fit_likelihood'].sort_values(ascending=False).index[:160]
scv.pl.heatmap(NF1_Ctrl_WT_velo, var_names=top_genes, sortby='velocity_pseudotime', col_color='labels_SCT', n_convolve=300,yticklabels=True, figsize = (8,32))


# In[192]:


top_genes = NF1_Ctrl_WT_velo2.var['fit_likelihood'].sort_values(ascending=False).index[:160]
scv.pl.heatmap(NF1_Ctrl_WT_velo2, var_names=top_genes, sortby='velocity_pseudotime', col_color='labels_SCT', n_convolve=300,yticklabels=True, figsize = (8,32))


# In[211]:


top_genes = NF1_Ctrl_WT_velo2.var['fit_likelihood'].sort_values(ascending=False).index[:160]
scv.pl.heatmap(NF1_Ctrl_WT_velo2, var_names=top_genes, sortby='velocity_pseudotime', col_color='labels_SCT', n_convolve=300,yticklabels=True, figsize = (8,32), save = 'Ctrl_heatmap_NF1.pdf')


# In[99]:


top_genes = NF1_OPG_NFhomo_velo.var['fit_likelihood'].sort_values(ascending=False).index[:160]
scv.pl.heatmap(NF1_OPG_NFhomo_velo, var_names=top_genes, sortby='velocity_pseudotime', col_color='labels_SCT', n_convolve=300,yticklabels=True, figsize = (8,32))


# In[193]:


top_genes = NF1_OPG_NFhomo_velo2.var['fit_likelihood'].sort_values(ascending=False).index[:160]
scv.pl.heatmap(NF1_OPG_NFhomo_velo2, var_names=top_genes, sortby='velocity_pseudotime', col_color='labels_SCT', n_convolve=300,yticklabels=True, figsize = (8,32))


# In[212]:


top_genes = NF1_OPG_NFhomo_velo2.var['fit_likelihood'].sort_values(ascending=False).index[:160]
scv.pl.heatmap(NF1_OPG_NFhomo_velo2, var_names=top_genes, sortby='velocity_pseudotime', col_color='labels_SCT', n_convolve=300,yticklabels=True, figsize = (8,32), save = 'OPG_heatmap_NF1.pdf')


# In[ ]:


NF1_Ctrl_WT_velo  NF1_OPG_NFhomo_velo


# In[159]:


NF1_total = NF1_Ctrl_WT_velo.concatenate([NF1_OPG_NFhomo_velo])


# In[161]:


sc.pp.neighbors(NF1_total)


# In[162]:


sc.tl.leiden(NF1_total)


# In[166]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_total, color = 'leiden', layout = 'fa', legend_loc = 'on data')


# In[170]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_total1, color = 'leiden', layout = 'fa', legend_loc = 'on data')


# In[169]:


NF1_total1 = NF1_total[NF1_total.obs['leiden'].isin(['0','4','12'])]


# In[174]:


NF1_total = NF1_total1


# In[177]:


NF1_Ctrl_WT_velo2 = NF1_total[NF1_total.obs['group_id'].isin(['Ctrl1_WT'])]
NF1_OPG_NFhomo_velo2 = NF1_total[NF1_total.obs['group_id'].isin(['OPG_NF1homo'])]


# In[172]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_Ctrl_WT_velo2, color = 'leiden', layout = 'fa')


# In[173]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_OPG_NFhomo_velo2, color = 'leiden', layout = 'fa')


# In[122]:


NF1_Ctrl_WT_velo = NF1_Ctrl_WT_velo[NF1_Ctrl_WT_velo.obs['leiden'].isin(['2','4','5','7'])]


# In[110]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_OPG_NFhomo_velo, color = 'leiden', layout = 'fa')


# In[111]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_Ctrl_WT_velo, color = 'leiden', layout = 'fa')


# In[175]:


pca_projections = pd.DataFrame(NF1_total.obsm["X_pca"],index=NF1_total.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections)
ms_data = palantir.utils.determine_multiscale_space(dm_res,n_eigs=4)
NF1_total.obsm["X_palantir"]=ms_data.values
sc.pp.neighbors(NF1_total,n_neighbors=30,use_rep="X_palantir")
NF1_total.obsm["X_pca2d"]=NF1_total.obsm["X_pca"][:,:2]
sc.tl.draw_graph(NF1_total,init_pos='X_pca2d')


# In[176]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_total, color = 'leiden', layout = 'fa')


# In[209]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_OPG_NFhomo_velo2, color = 'Siglec1', layout = 'fa', use_raw = False)


# In[210]:


sc.set_figure_params(figsize = [6,6])
sc.pl.draw_graph(NF1_Ctrl_WT_velo2, color = 'Siglec1', layout = 'fa', use_raw = False)


# In[180]:


scv.pp.filter_and_normalize(NF1_OPG_NFhomo_velo2, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(NF1_OPG_NFhomo_velo2, n_pcs=30, n_neighbors=30)
scv.tl.velocity(NF1_OPG_NFhomo_velo2)
scv.tl.velocity_graph(NF1_OPG_NFhomo_velo2)


# In[182]:


sc.pp.neighbors(NF1_Ctrl_WT_velo2)
scv.pp.filter_and_normalize(NF1_Ctrl_WT_velo2, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(NF1_Ctrl_WT_velo2, n_pcs=30, n_neighbors=30)
scv.tl.velocity(NF1_Ctrl_WT_velo2)
scv.tl.velocity_graph(NF1_Ctrl_WT_velo2)


# In[181]:


scv.pl.velocity_embedding_grid(NF1_OPG_NFhomo_velo2, basis='draw_graph_fa', color = 'labels_SCT', dpi = 300, add_margin = 0.2, figsize = (4,4), arrow_length = 4, arrow_size = 2, density = 1)


# In[183]:


scv.pl.velocity_embedding_grid(NF1_Ctrl_WT_velo2, basis='draw_graph_fa', color = 'labels_SCT', dpi = 300, add_margin = 0.2, figsize = (4,4), arrow_length = 4, arrow_size = 2, density = 1)


# In[184]:


scv.tl.velocity_pseudotime(NF1_Ctrl_WT_velo2)
scv.pl.scatter(NF1_Ctrl_WT_velo2, color='velocity_pseudotime', cmap='gnuplot', basis='draw_graph_fa')


# In[185]:


scv.tl.velocity_pseudotime(NF1_OPG_NFhomo_velo2)
scv.pl.scatter(NF1_OPG_NFhomo_velo2, color='velocity_pseudotime', cmap='gnuplot', basis='draw_graph_fa')


# In[ ]:





# In[ ]:





# In[131]:


cluster_small_multiples(NF1_pop, clust_key = 'group_id', save = 'NF1_other.pdf')


# In[132]:


cluster_small_multiples(NF1_pop, clust_key = 'group_id')


# In[135]:


NF1_pop.obs


# In[144]:


sc.set_figure_params()
sc.tl.rank_genes_groups(NF1, 'labels_SCT', method='logreg')
sc.pl.rank_genes_groups(NF1, n_genes=25, sharey=False)


# In[141]:


NF1_pop.raw = NF1_pop


# In[145]:


def cluster_small_multiples(adata, clust_key, size=60, frameon=False, legend_loc=None, **kwargs):
    tmp = adata.copy()

    for i,clust in enumerate(adata.obs[clust_key].cat.categories):
        tmp.obs[clust] = adata.obs[clust_key].isin([clust]).astype('category')
        tmp.uns[clust+'_colors'] = ['#d3d3d3', adata.uns[clust_key+'_colors'][i]]

    sc.pl.draw_graph(tmp, groups=tmp.obs[clust].cat.categories[1:].values, color=adata.obs[clust_key].cat.categories.tolist(), size=size, frameon=frameon, legend_loc=legend_loc, **kwargs)


# In[146]:


cluster_small_multiples(NF1, clust_key = 'labels_SCT')


# In[4]:


NF1_new = NF1[NF1.obs['group_id'].isin(['Ctrl2_NF1floxed','OPG_NF1homo'])]


# In[5]:


sc.pl.draw_graph(NF1_new,color=["labels_SCT"], legend_loc = 'right margin')


# In[6]:


NF1_new = NF1[NF1.obs['labels_SCT'].isin(['Olig_1','OPC','Neuron?','Astro_1','Astro_2','Astro_3','Astro_4',
                                          'Endothelia','Fibro_1','Fibro_2','Macrophage','Microglia','Mural','Pericyte',
                                         'T_cell','Unknown_1','Unknown_2','Unknown_3'])]


# In[7]:


sc.pl.draw_graph(NF1_new,color=["labels_SCT"], legend_loc = 'right margin')


# In[8]:


sc.pl.umap(NF1_new,color=["labels_SCT"], legend_loc = 'right margin')


# In[9]:


sc.tl.umap(NF1_new)


# In[10]:


sc.pl.umap(NF1_new,color=["labels_SCT"], legend_loc = 'right margin')


# In[11]:


pca_projections = pd.DataFrame(NF1_new.obsm["X_pca"],index=NF1_new.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections)
ms_data = palantir.utils.determine_multiscale_space(dm_res,n_eigs=4)
NF1_new.obsm["X_palantir"]=ms_data.values
sc.pp.neighbors(NF1_new,n_neighbors=30,use_rep="X_palantir")
NF1_new.obsm["X_pca2d"]=NF1_new.obsm["X_pca"][:,:2]
sc.tl.draw_graph(NF1_new,init_pos='X_pca2d')


# In[12]:


sc.pl.draw_graph(NF1_new,color=["labels_SCT"], legend_loc = 'right margin')


# In[13]:


NF1_new


# In[15]:


sc.pl.draw_graph(NF1_new,color=["leiden",'group_id'], legend_loc = 'right margin')


# In[ ]:




