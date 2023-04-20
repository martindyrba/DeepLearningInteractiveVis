#####
# Read the data
#
# read the Excel sheet containing data
library(readxl)
Hippocampus_values <- as.data.frame(read_excel("hippocampus_volume_relevance_DELCODE.xlsx", sheet="DELCODE_LRP_CMP"))
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="prmdiag"] <- "Group"
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="age"] <- "Age at scan"
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="FieldStrength"] <-"MRI_Field_Strength"
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="sex_bin_1female"] <-"Sex (1=female)"
Hippocampus_values$Group <- factor(Hippocampus_values$Group, levels=c(0,2,5), labels=c("CN", "MCI", "AD"))
#View(Hippocampus_values)
# rescale volume from mm³ to ml; the same units as TIV
Hippocampus_values$aal_hippocampus <- Hippocampus_values$aal_hippocampus / 1000
# add columns brain excluding hippocampus or temporal lobe
Hippocampus_values$sum_relevance_brain_wo_hippocampus <- Hippocampus_values$sum_relevance_brain - Hippocampus_values$sum_relevance_hippocampus
Hippocampus_values$sum_relevance_brain_wo_temporal <- Hippocampus_values$sum_relevance_brain - Hippocampus_values$sum_relevance_temporal

#####
# residualize the hippocampus volumes with same covariates as used for the gray matter maps before
load(file='linear_model_hipp_vol_ADNI2.RData')

# estimate residuals for all cases
hipp_vol_res <- Hippocampus_values$aal_hippocampus - predict(hp_lm, newdata = Hippocampus_values)
write.csv(cbind(Hippocampus_values$SID, hipp_vol_res), file="residualized_vol_DELCODE_n474.csv", row.names=F)


#####
# get correlation of hippocampus volume residuals and hippocampus LRP values

# first, remove outliers from all "Sum_relevance_*" columns, i.e. set those values to NA
#  adapted from https://statisticsglobe.com/remove-outliers-from-data-set-in-r
remove_outliers <- function(dat, cols) {
  for (c in cols) {
    dat[dat[,c] %in% boxplot.stats(dat[,c])$out, c] <- NA
  }
  return(dat)
}
Hippocampus_values <- remove_outliers(Hippocampus_values, grep("Sum_relevance", colnames(Hippocampus_values)))

print(cor(hipp_vol_res, Hippocampus_values$sum_relevance_hippocampus, use="pairwise.complete.obs"))
  correls_cv_hipp <- sapply(grep("Sum_relevance_hippocampus_cv", colnames(Hippocampus_values)), function(x) do.call(cor, list(x=Hippocampus_values[,x], y=hipp_vol_res, use="pairwise.complete.obs")))
  summary(correls_cv_hipp)
print(cor(hipp_vol_res, Hippocampus_values$sum_relevance_brain, use="pairwise.complete.obs"))
print(cor(Hippocampus_values$sum_relevance_hippocampus, Hippocampus_values$sum_relevance_brain_wo_hippocampus, use="pairwise.complete.obs"))
print(cor(Hippocampus_values$sum_relevance_hippocampus, Hippocampus_values$sum_relevance_brain_wo_temporal, use="pairwise.complete.obs"))
print(cor(Hippocampus_values$sum_relevance_brain, Hippocampus_values$sum_relevance_brain_wo_hippocampus, use="pairwise.complete.obs"))
print(cor(hipp_vol_res, Hippocampus_values$sum_relevance_brain_wo_hippocampus, use="pairwise.complete.obs"))
print(cor(hipp_vol_res, Hippocampus_values$sum_relevance_brain_wo_temporal, use="pairwise.complete.obs"))
  correls_cv_brhipp <- sapply(grep("Sum_relevance_brain_cv", colnames(Hippocampus_values)), function(x) do.call(cor, list(x=Hippocampus_values[,x], y=hipp_vol_res, use="pairwise.complete.obs")))
  summary(correls_cv_brhipp)

# scatter plot bilateral vol vs. rel
par(mar=c(4,4,1,0), oma=c(0,0,0,0))
pchs<-c(16, 15, 17)
mycolors <- c("#111111", "#61D04F", "#2297E6")

lm_both <- lm(hipp_vol_res~Hippocampus_values$sum_relevance_hippocampus) # keeps scale and intercept
cor_both <- cor.test(hipp_vol_res, Hippocampus_values$sum_relevance_hippocampus) # same as beta of lm(scale(vol_res),scale(act))
plot(jitter(Hippocampus_values$sum_relevance_hippocampus, factor=200), hipp_vol_res,
     cex = 0.7, pch=pchs[Hippocampus_values$Group], col=mycolors[Hippocampus_values$Group],
     axes = FALSE,
     xlab = "Total relevance of hippocampus voxels", ylab = "Bilateral hippocampus volume (residuals, ml)",
     main = "Correlation of hippocampus volume and CNN relevance scores")
legend(x = "topright", legend=c("Normal", "MCI", "AD"), col=mycolors, pch=c(16,15,17), bty="n")
axis(1, at=NULL, labels=T)
axis(2, at=NULL, labels=T)
abline(a=lm_both$coefficients[1], b=lm_both$coefficients[2], col="red")
text(x=1, y=-4.1, labels=paste0("r = ", round(cor_both$estimate, digits=2),
                                ", p ", ifelse(cor_both$p.value<0.001, "< 0.001", paste0("= ", round(cor_both$estimate, digits=2))),
                                ", R² = ", round(summary(lm_both)$r.squared, digits=2),
                                ", n = ", cor_both$parameter+2), col="red")

# scatter plot of HP vol vs. rel brain
lm_both <- lm(hipp_vol_res~Hippocampus_values$sum_relevance_brain) # keeps scale and intercept
cor_both <- cor.test(hipp_vol_res, Hippocampus_values$sum_relevance_brain) # same as beta of lm(scale(vol_res),scale(act))
plot(jitter(Hippocampus_values$sum_relevance_brain, factor=200), hipp_vol_res,
     cex = 0.7, pch=pchs[Hippocampus_values$Group], col=mycolors[Hippocampus_values$Group],
     axes = FALSE,
     xlab = "Total relevance of gray matter voxels", ylab = "Bilateral hippocampus volume (residuals, ml)",
     main = "Correlation of hippocampus volume and CNN gray matter relevance scores")
legend(x = "topright", legend=c("Normal", "MCI", "AD"), col=c(1,3,4), pch=c(16,15,17), bty="n")
axis(1, at=NULL, labels=T)
axis(2, at=NULL, labels=T)
abline(a=lm_both$coefficients[1], b=lm_both$coefficients[2], col="red")
text(x=13, y=-4.1, labels=paste0("r = ", round(cor_both$estimate, digits=2),
                                 ", p ", ifelse(cor_both$p.value<0.001, "< 0.001", paste0("= ", round(cor_both$estimate, digits=2))),
                                 ", R² = ", round(summary(lm_both)$r.squared, digits=2),
                                 ", n = ", cor_both$parameter+2), col="red")

# scatter plot of HP vol vs. rel brain w/o hippocampus
lm_both <- lm(hipp_vol_res~Hippocampus_values$sum_relevance_brain_wo_hippocampus) # keeps scale and intercept
cor_both <- cor.test(hipp_vol_res, Hippocampus_values$sum_relevance_brain_wo_hippocampus) # same as beta of lm(scale(vol_res),scale(act))
plot(jitter(Hippocampus_values$sum_relevance_brain_wo_hippocampus, factor=200), hipp_vol_res,
     cex = 0.7, pch=pchs[Hippocampus_values$Group], col=mycolors[Hippocampus_values$Group],
     axes = FALSE,
     xlab = "Total relevance of gray matter voxels excluding hippocampus", ylab = "Bilateral hippocampus volume (residuals, ml)",
     main = "Correlation of hippocampus volume and CNN gray matter relevance scores excluding hippocampus")
legend(x = "topright", legend=c("Normal", "MCI", "AD"), col=c(1,3,4), pch=c(16,15,17), bty="n")
axis(1, at=NULL, labels=T)
axis(2, at=NULL, labels=T)
abline(a=lm_both$coefficients[1], b=lm_both$coefficients[2], col="red")
text(x=13, y=-4.1, labels=paste0("r = ", round(cor_both$estimate, digits=2),
                                 ", p ", ifelse(cor_both$p.value<0.001, "< 0.001", paste0("= ", round(cor_both$estimate, digits=2))),
                                 ", R² = ", round(summary(lm_both)$r.squared, digits=2),
                                 ", n = ", cor_both$parameter+2), col="red")

# scatter plot of HP vol vs. rel brain w/o temporal lobe
lm_both <- lm(hipp_vol_res~Hippocampus_values$sum_relevance_brain_wo_temporal) # keeps scale and intercept
cor_both <- cor.test(hipp_vol_res, Hippocampus_values$sum_relevance_brain_wo_temporal) # same as beta of lm(scale(vol_res),scale(act))
plot(jitter(Hippocampus_values$sum_relevance_brain_wo_temporal, factor=200), hipp_vol_res,
     cex = 0.7, pch=pchs[Hippocampus_values$Group], col=mycolors[Hippocampus_values$Group],
     axes = FALSE,
     xlab = "Total relevance of gray matter voxels excluding temporal lobe", ylab = "Bilateral hippocampus volume (residuals, ml)",
     main = "Correlation of hippocampus volume and CNN gray matter relevance scores excluding temporal lobe")
legend(x = "topright", legend=c("Normal", "MCI", "AD"), col=c(1,3,4), pch=c(16,15,17), bty="n")
axis(1, at=NULL, labels=T)
axis(2, at=NULL, labels=T)
abline(a=lm_both$coefficients[1], b=lm_both$coefficients[2], col="red")
text(x=8, y=-4.1, labels=paste0("r = ", round(cor_both$estimate, digits=2),
                              ", p ", ifelse(cor_both$p.value<0.001, "< 0.001", paste0("= ", round(cor_both$estimate, digits=2))),
                              ", R² = ", round(summary(lm_both)$r.squared, digits=2),
                              ", n = ", cor_both$parameter+2), col="red")


# create new data frame for all values
dat <- data.frame("RID"=Hippocampus_values$SID,
                  "HP_vol_res"=hipp_vol_res,
                  "LRP_hipp"=Hippocampus_values$sum_relevance_hippocampus,
                  "LRP_brain"=Hippocampus_values$sum_relevance_brain,
                  #"LRP_brain_w/o_hipp"=Hippocampus_values$sum_relevance_brain_wo_hippocampus,
                  #"LRP_brain_w/o_temporal"=Hippocampus_values$sum_relevance_brain_wo_temporal,
                  "LRP_temporal"=Hippocampus_values$sum_relevance_temporal,
                  "LRP_occipital"=Hippocampus_values$sum_relevance_occipital,
                  "LRP_frontal"=Hippocampus_values$sum_relevance_frontal,
                  "LRP_parietal"=Hippocampus_values$sum_relevance_parietal,
                  "LRP_insula_cingulate"=Hippocampus_values$sum_relevance_insula_cingulate,
                  "LRP_basal_ganglia"=Hippocampus_values$sum_relevance_basal_ganglia,
                  "LRP_cerebellum"=Hippocampus_values$sum_relevance_cerebellum,
                  "group"=Hippocampus_values$Group,
                  "Amy_status"=Hippocampus_values$ratio_Abeta42_40_pos,
                  "group_bin"=(Hippocampus_values$Group!="CN")+0)


#####
# create correlation matrix plots to evaluate relevance location and sensitivity for hippocampus volume
library(Hmisc)
library(corrplot)

dat2 <- subset(dat, select = grep("LRP_",colnames(dat)))
dat2 <- cbind(unclass(dat$group), dat$HP_vol_res, dat2)
colnames(dat2)[c(1,2)] <- c("Group", "Hipp_vol_res")
res2 <- rcorr(data.matrix(dat2))

par(oma=c(0,0,2.1,0), xpd=NA)
corrplot(res2$r, p.mat = res2$P, type = "upper", tl.col = "black", tl.srt = 45, sig.level = 0.001, insig = "blank", tl.pos = "lt")
corrplot(res2$r, add = TRUE, type = "lower", method = "number", col = "black", diag = FALSE, tl.pos = "n", cl.pos = "n", number.cex=0.8)


#####
# obtain AUC and accuracy for hippocampus volume residuals

require(pROC)
# helper function to obtain values for respective group (MCI or AD) and threshold
get.values <- function(thr, grp) {
  # get confusion matrix for test data using threshold
  testdat <- subset(dat, subset=((dat$group=="CN" | dat$group==grp)))
  testdat$pred <- (testdat$HP_vol_res<thr)+0
  confmat <- xtabs(~group+pred, data=testdat)[c("CN",grp),] # subset(testdat, subset=testdat$Amy_status==testdat$group_bin)) # mimic CNN analysis were model is estimated also including all people, and amyloid stratification being applied post-hoc
  stopifnot(nrow(confmat)==2, ncol(confmat)==2)
  tn <- confmat[1,1]; fp <- confmat[1,2]; fn <- confmat[2,1]; tp <- confmat[2,2]
  sen = tp / (tp+fn)
  spec = tn / (fp+tn)
  ppv = tp / (tp+fp)
  npv = tn / (tn+fn)
  f1 = 2 * ((ppv * sen) / (ppv + sen))
  bacc = (spec + sen) / 2
  # additionally calculate AUC for test data
  auc.test <- auc(roc(group~HP_vol_res, levels=c("CN",grp), direction=">", data=testdat)) # subset(testdat, subset=testdat$Amy_status==testdat$group_bin))) # mimic CNN analysis were model is estimated also including all people, and amyloid stratification being applied post-hoc
  # return result metrics
  return(data.frame(Threshold=thr, AUC=as.numeric(auc.test), Balanced_Accuracy=bacc, Sensitivity=sen, Specificity=spec, PPV=ppv, NPV=npv, F1=f1))
}

par(oma=c(0,0,0,0), xpd=F)

## apply thresholds from ADNI2

# AUC for hippocampus residuals for whole sample
myroc.ad <- roc(group~HP_vol_res, levels=c("CN","AD"), direction=">",
                data=subset(dat,subset=(dat$group!="MCI") ))
plot(myroc.ad, col=mycolors[3])
text(x=0, y=0.05, labels=paste0("AD: AUC = ", round(auc(myroc.ad), digits=3)), col=mycolors[3])
print(auc(myroc.ad))
#print(ci.auc(myroc.ad, method='bootstrap'))
results.ad <- do.call(rbind, lapply(list(-0.9460807), get.values, grp="AD"))
print(results.ad)
results.ad <- do.call(rbind, lapply(c(-0.9471819, -0.6764138, -0.9939213, -0.9460807, -0.9460807, -0.9460807, -0.6764138, -0.7890531, -0.676413, -0.9460807), get.values, grp="AD"))
results.mean.ad <- cbind(colMeans(results.ad), apply(results.ad, 2, sd))
colnames(results.mean.ad) <- c("mean", "sd")
print(results.mean.ad)

myroc.mci <- roc(group~HP_vol_res, levels=c("CN","MCI"), direction=">",
                 data=subset(dat,subset=(dat$group!="AD") ))# & (dat$Amy_status==dat$group_bin)))
plot(myroc.mci, col=mycolors[2], add=T)
text(x=0, y=0, labels=paste0("MCI: AUC = ", round(auc(myroc.mci), digits=3)), col=mycolors[2])
legend(x = "topright", legend=c("AD","MCI"), col=mycolors[c(3,2)], pch=15, bty="n")
print(auc(myroc.mci))
#print(ci.auc(myroc.mci, method='bootstrap'))
results.mci <- do.call(rbind, lapply(list(-0.6306185), get.values, grp="MCI"))
print(results.mci)
results.mci <- do.call(rbind, lapply(c(-0.6666399, -0.6306185, -0.7923484, -0.5286460, -0.5286460, -0.6306185, -0.6306185, -0.6306185, -0.6306185, -0.6446721), get.values, grp="MCI"))
results.mean.mci <- cbind(colMeans(results.mci), apply(results.mci, 2, sd))
colnames(results.mean.mci) <- c("mean", "sd")
print(results.mean.mci)



#####
# generate violin plots for hippocampus volume residuals

library(ggplot2)
mycolors <- c("gray", "#61D04F", "#2297E6")

# group separation by hippocampus volume (residualized)
p <- ggplot(dat, aes(x=factor(group, labels=c("CN", "MCI", "AD")), y=HP_vol_res)) +
  geom_violin() +
  geom_hline(aes(yintercept=results.mci[1,1], color="red"), linetype="dashed") +
  geom_hline(aes(yintercept=results.ad[1,1], color="red")) +
  labs(x="Groups", y="Bilateral hippocampus volume (residuals, ml)", title="Distribution of hippocampus volume") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_boxplot(width=0.2, aes(fill=factor(group))) +
  scale_fill_manual(values=mycolors) +
  theme(legend.position="none") +
  annotate(geom="text", x=3.2, y=1.8, label=paste0("- Threshold for AD: < ",round(results.ad[1,1], digits=3)), color="red") +
  annotate(geom="text", x=3.2, y=2, label=paste0("- - Threshold for MCI: < ",round(results.mci[1,1], digits=3)), color="red")
plot(p)


#####
# optionally, obtain group separation by hippocampus LRP values

# group separation by hippocampus relevance scores --> comparable to raw CNN AUC for that specific model/CV-fold
p2 <- ggplot(dat, aes(x=factor(group, labels=c("CN", "MCI", "AD")), y=LRP_hipp)) +
  geom_violin() +
  #geom_hline(aes(yintercept=thr[1], color="red")) +
  labs(x="Groups", y="Bilateral hippocampus relevance scores (log10)") +
  theme(legend.position="none") +
  scale_y_continuous(trans='log10') +
  geom_boxplot(width=0.2, aes(fill=factor(group))) +
  scale_fill_manual(values=mycolors) +
  theme(legend.position="none")
plot(p2)
