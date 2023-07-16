retina_NF1 <- readRDS("~/10X/NF1_retina/MouseNF1_retina_RGCs_cluster2.rds")
load("~/10X/NF1_retina/OPG_data_clean_sct_25PCA.RData")
load("~/sampleComparisonEnv.RData")

retina_NF1$origtissue <- 'Retina'
OPG_data_clean_sct$origtissue <- 'ON'
ProcessInt <- function(data.integrated){
  data.integrated <- RunPCA(data.integrated, npcs = 30, verbose = FALSE)
  data.integrated <- FindNeighbors(data.integrated, dims = 1:20)
  data.integrated <- FindClusters(data.integrated, resolution = 0.5)
  data.integrated <- RunUMAP(data.integrated, reduction = "pca", dims = 1:20)
  data.integrated <- RunTSNE(data.integrated,  dims.use = 1:20 )
}
integration_list <- list(retina_NF1,OPG_data_clean_sct)
features <- SelectIntegrationFeatures(object.list = integration_list)
integration_list <- PrepSCTIntegration(integration_list, anchor.features = features)
data.anchors <- FindIntegrationAnchors(object.list = integration_list, anchor.features = features, normalization.method = 'SCT')
data.combined <- IntegrateData(anchorset = data.anchors, normalization.method = 'SCT')
shuhong.combined <- ProcessInt(data.combined)

DimPlot(shuhong.combined, group.by = 'EK_anno', label = T, repel = T, label.box = T)
shuhong.combined <- subset(shuhong.combined, subset = EK_anno == c('ON Olig_2'), invert = T)
DimPlot(shuhong.combined, group.by = 'EK_anno', label = T, repel = T, label.box = T)

table(shuhong.combined$orig.ident)
head(OPG_data_clean_sct)
head(retina_NF1)
table(retina_NF1$genetics_age)

adult <- readRDS('/home/baranov_lab/mouseM1.rds')
adult$orig.ident <- 'adult_wt_retina'
table(adult@active.ident)
adult$EK_anno <- adult@active.ident


library(sctransform)
head(adult)
adult <- SCTransform(adult, vars.to.regress = c('nCount_RNA', 'nFeature_RNA','percent.rb','percent.mt','S.Score','G2M.Score'), verbose = TRUE)
DefaultAssay(adult) <- 'SCT'
ProcessSeu <- function(Seurat){

  Seurat <- RunPCA(Seurat)
  Seurat <- FindNeighbors(Seurat, dims = 1:10)
  Seurat <- FindClusters(Seurat, resolution = 0.5)
  Seurat <- RunUMAP(Seurat, dims = 1:10)
  Seurat <- RunTSNE(Seurat,  dims.use = 1:10 )
  DimPlot(object = Seurat, reduction = "umap")
  return (Seurat)
}
adult <- ProcessSeu(adult)
DimPlot(adult, group.by = 'EK_anno', label = T, repel = T)


integration_list <- list(retina_NF1,OPG_data_clean_sct, adult)
retina_NF1 <- PrepSCTFindMarkers(object = retina_NF1)
OPG_data_clean_sct <- PrepSCTFindMarkers(object = OPG_data_clean_sct)
features <- SelectIntegrationFeatures(object.list = integration_list, nfeatures = 5000)
integration_list <- PrepSCTIntegration(integration_list, anchor.features = features)
data.anchors <- FindIntegrationAnchors(object.list = integration_list, anchor.features = features, normalization.method = 'SCT')
data.combined <- IntegrateData(anchorset = data.anchors, normalization.method = 'SCT')
shuhong.combined <- ProcessInt(data.combined)
shuhong.combined$subsetinfo <- paste(shuhong.combined$orig.ident, shuhong.combined$age)


OPG_data <- subset(shuhong.combined, subset = subsetinfo == c('OPG_NF1homo_sample1 NA','OPG_NF1homo_sample2 NA','OPG_NF1homo_sample3 NA',
                                                              'OPG_NF1homo_sample4 NA','NF1RGCFMC1 8mo','NF1RGCFMC2 8mo','NF1MICFMC 8mo'))

twoflox <- subset(shuhong.combined, subset = subsetinfo == c('Ctrl2_NF1floxed_sample1 NA','Ctrl2_NF1floxed_sample2 NA','NF1RGCFF1 8mo','NF1MICFF 8mo'))

wtwt <- subset(shuhong.combined, subset = subsetinfo == c('Ctrl1_WT_sample2 NA','Ctrl1_WT_sample1 NA', 'adult NA'))
                                                          
hetero <- subset(shuhong.combined, subset = subsetinfo == c('Ctrl3_NF1hetero_sample1 NA','Ctrl3_NF1hetero_sample2 NA','NF1RGCHT 8mo'))


samples.combined <- RenameIdents(samples.combined, '0'=	'Fibroblasts 1',
                                 '1'=	'Macrophage 1',
                                 '2'=	'Myelinating SC 1',
                                 '3'=	'Neurons 1',
                                 '4'=	'Fibroblasts 2',
                                 '5'=	'Non-myelinating SC',
                                 '6'=	'Macrophage 2',
                                 '7'=	'Neurons 1',
                                 '8'=	'Satellite cells',
                                 '9'=	'Macrophage 2',
                                 '10'=	'Macrophage 3',
                                 '11'=	'Myelinating SC 2',
                                 '12'=	'Perineurial cells',
                                 '13'=	'Endothelial cells',
                                 '14'=	'Pericytes',
                                 '15'=	'Neurons 2',
                                 '16'=	'Pericytes',
                                 '17'=	'Endoneurial fibroblasts 1',
                                 '18'=	'Endoneurial fibroblasts 2',
                                 '19'=	'Neurons 3',
                                 '20'=	'Neurons 4',
                                 '21'=	'SC precursors',
                                 '22'=	'T cells',
                                 '23'=	'Macrophage 4',
                                 '24'=	'Perineurial cells',
                                 '25'=	'Fibroblasts 1',
                                 '26'=	'MDSCs',
                                 '27'=	'Neurons 5',
                                 '28'=	'Neurons 6',
                                 '29'=	'Neurons 4',
                                 '30'=	'Proliferating cells'
)

samples.combined$EK_anno <- samples.combined@active.ident
plneu2mo <- subset(samples.combined, subset = orig.ident == '2moDhhPlus')
plneu7mo <- subset(samples.combined, subset = orig.ident == '7moDhhPlus')
plneu2moNF <- subset(samples.combined, subset = orig.ident == '10XRatner-Nf2832-2months')
plneu7moNF <- subset(samples.combined, subset = orig.ident == '10XRatner-Nf2617-7months')


library(CellChat)

runCellChat <- function(object, group.by, output_path) {
  # Create cellchat object
  cellchat <- createCellChat(object = object, group.by = group.by)
  
  # Set CellChatDB to mouse
  CellChatDB <- CellChatDB.mouse
  CellChatDB.use <- CellChatDB
  cellchat@DB <- CellChatDB.use
  
  # Subset data
  cellchat <- subsetData(cellchat)
  
  # Identify overexpressed genes
  cellchat <- identifyOverExpressedGenes(cellchat)
  
  # Identify overexpressed interactions
  cellchat <- identifyOverExpressedInteractions(cellchat)
  
  # Project data using mouse PPI
  cellchat <- projectData(cellchat, PPI.mouse)
  
  # Compute communication probabilities
  cellchat <- computeCommunProb(cellchat, trim = 0.25, population.size = FALSE, raw.use = FALSE)
  
  # Filter communication based on minimum cell count
  cellchat <- filterCommunication(cellchat, min.cells = 10)
  
  # Compute communication probabilities for pathways
  cellchat <- computeCommunProbPathway(cellchat)
  
  # Aggregate network
  cellchat <- aggregateNet(cellchat)
  
  # Compute centrality
  cellchat <- netAnalysis_computeCentrality(cellchat, slot.name = "netP")
  
  # Generate signaling role heatmaps
  ht1 <- netAnalysis_signalingRole_heatmap(cellchat, pattern = "outgoing", height = 30)
  ht2 <- netAnalysis_signalingRole_heatmap(cellchat, pattern = "incoming", height = 30)
  
  # Combine heatmaps
  heatmap_combined <- ht1 + ht2
  
  # Save cellchat object as RDS file
  saveRDS(cellchat, output_path)
}

# Run cellchat analysis for twoflox
runCellChat(object = twoflox, group.by = "EK_anno", output_path = "/home/baranov_lab/10X/NF1/cellchat/cellchat_twoflox.rds")

# Run cellchat analysis for wtwt
runCellChat(object = wtwt, group.by = "EK_anno", output_path = "/home/baranov_lab/10X/NF1/cellchat/cellchat_wtwt.rds")

# Run cellchat analysis for OPG_data
runCellChat(object = OPG_data, group.by = "EK_anno", output_path = "/home/baranov_lab/10X/NF1/cellchat/cellchat_OPG_data.rds")
