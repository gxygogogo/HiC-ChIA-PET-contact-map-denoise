library(pheatmap)
library(ComplexHeatmap)

## Enhanced
df = read.table("/public1/xinyu/CohesinProject/SRHiC/predict/WT_SMC1A/5K_removeDiagonal/enhanced_chr16_remove_Diagonal.totalMats.txt", sep = "\t", header = F)
df = log10(df + 1)

## raw data
# df = read.table("/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_Mat/chr16.totalMats.txt", sep = "\t", header = F)
# df = log10(df + 1)

# chr16:54M-60M(chr16:54800000-60400000)
df_54_60 = df[10960:12080,10960:12080]
pdf("/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_Mat/WT_SMC1A/5K_removeDiagonal/chr16_54M-60M.pdf", width = 10, height = 10)
Heatmap(df_54_60,
        cluster_rows = F,
        cluster_columns = F,
        show_row_names = F,
        show_column_names = F,
        border = TRUE,
        heatmap_legend_param = list(title = "Value"),
        col = colorRampPalette(colors = c("white","white","#DA3433","#D81115"))(200)
        )
dev.off()

# chr16:64M-70M(chr16:64800000-70400000)
df_64_70 = df[12960:14080,12960:14080]
pdf("/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_Mat/WT_SMC1A/5K_removeDiagonal/chr16_64M-70M.pdf", width = 10, height = 10)
Heatmap(df_64_70,
        cluster_rows = F,
        cluster_columns = F,
        show_row_names = F,
        show_column_names = F,
        border = TRUE,
        heatmap_legend_param = list(title = "Value"),
        col = colorRampPalette(colors = c("white","white","#DA3433","#D81115"))(200)
        )
dev.off()

# chr16:57M-58M(chr16:57000000-58000000)
df_57_58 = df[11400:11600,11400:11600]
pdf("/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_Mat/WT_SMC1A/5K_removeDiagonal/chr16_57M-58M.pdf", width = 10, height = 10)
Heatmap(df_57_58,
        cluster_rows = F,
        cluster_columns = F,
        show_row_names = F,
        show_column_names = F,
        border = TRUE,
        heatmap_legend_param = list(title = "Value"),
        col = colorRampPalette(colors = c("white","white","#DA3433","#D81115"))(200)
        )
dev.off()

# chr16:48M-49M(chr16:48000000-49000000)
df_48_49 = df[9600:9800,9600:9800]
pdf("/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_Mat/WT_SMC1A/5K_removeDiagonal/chr16_48M-49M.pdf", width = 10, height = 10)
Heatmap(df_48_49,
        cluster_rows = F,
        cluster_columns = F,
        show_row_names = F,
        show_column_names = F,
        border = TRUE,
        heatmap_legend_param = list(title = "Value"),
        col = colorRampPalette(colors = c("white","white","#DA3433","#D81115"))(200)
        )
dev.off()

# chr16:50M-50.5M(chr16:50000000-50500000)
df_50_half = df[5000:5050,5000:5050]
pdf("/public1/xinyu/CohesinProject/SRHiC/predict/WT_SMC1A/5K_removeDiagonal/chr16_50_half.pdf", width = 10, height = 10)
Heatmap(df_50_half,
        cluster_rows = F,
        cluster_columns = F,
        show_row_names = F,
        show_column_names = F,
        border = TRUE,
        heatmap_legend_param = list(title = "Value"),
        col = colorRampPalette(colors = c("white","white","#DA3433","#D81115"))(200)
        )
dev.off()

# chr16:67M-67.5M(chr16:67000000-67500000)
df_67_half = df[6700:6750,6700:6750]
pdf("/public1/xinyu/CohesinProject/SRHiC/predict/WT_SMC1A/5K_removeDiagonal/chr16_67_half.pdf", width = 10, height = 10)
Heatmap(df_67_half,
        cluster_rows = F,
        cluster_columns = F,
        show_row_names = F,
        show_column_names = F,
        border = TRUE,
        heatmap_legend_param = list(title = "Value"),
        col = colorRampPalette(colors = c("white","white","#DA3433","#D81115"))(200)
        )
dev.off()


