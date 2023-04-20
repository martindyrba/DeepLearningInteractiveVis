#####
# Read the data
#
# read the Excel sheet containing data
library(readxl)
Hippocampus_values <- read_excel("aibl_ptdemog_final.xlsx", sheet="aibl_ptdemog_final")
stopifnot(nrow(Hippocampus_values)==606)

# adjust column names to match those in ADNI2 linear model
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="DXCURREN"] <- "Group"
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="age"] <- "Age at scan"
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="field_strength"] <-"MRI_Field_Strength"
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="PTGENDER(1=Female)"] <-"Sex (1=female)"
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="Total"] <-"TIV_CAT12"
Hippocampus_values$group_bin <- (Hippocampus_values$Group>1)+0
Hippocampus_values$Group <- factor(Hippocampus_values$Group, levels=c(1,2,3), labels=c("CN", "MCI", "AD"))
colnames(Hippocampus_values)[colnames(Hippocampus_values)=="conversion_CL_threshold_26"] <-"Amy_status"
#View(Hippocampus_values)
# rescale volume from mm³ to ml; the same units as TIV
Hippocampus_values$aal_hippocampus <- Hippocampus_values$aal_hippocampus / 1000

#####
# residualize the hippocampus volumes with same covariates as used for the gray matter maps before
load(file='linear_model_hipp_vol_ADNI2.RData')

# estimate residuals for all cases
hipp_vol_res <- Hippocampus_values$aal_hippocampus - predict(hp_lm, newdata = Hippocampus_values)
Hippocampus_values$HP_vol_res <- hipp_vol_res


require(pROC)
# helper function to obtain values for respective group (MCI or AD) and threshold
get.values <- function(thr, grp) {
  # get confusion matrix for test data using threshold
  testdat <- subset(Hippocampus_values, subset=((Hippocampus_values$Group=="CN" | Hippocampus_values$Group==grp)))
  testdat$pred <- (testdat$HP_vol_res<thr)+0
  confmat <- xtabs(~Group+pred, data=testdat)[c("CN",grp),] # subset(testdat, subset=testdat$Amy_status==testdat$group_bin)) # mimic CNN analysis were model is estimated also including all people, and amyloid stratification being applied post-hoc
  stopifnot(nrow(confmat)==2, ncol(confmat)==2)
  tn <- confmat[1,1]; fp <- confmat[1,2]; fn <- confmat[2,1]; tp <- confmat[2,2]
  sen = tp / (tp+fn)
  spec = tn / (fp+tn)
  ppv = tp / (tp+fp)
  npv = tn / (tn+fn)
  f1 = 2 * ((ppv * sen) / (ppv + sen))
  bacc = (spec + sen) / 2
  # additionally calculate AUC for test data
  auc.test <- auc(roc(Group~HP_vol_res, levels=c("CN",grp), direction=">", data=testdat)) # subset(testdat, subset=testdat$Amy_status==testdat$group_bin))) # mimic CNN analysis were model is estimated also including all people, and amyloid stratification being applied post-hoc
  # return result metrics
  return(data.frame(Threshold=thr, AUC=as.numeric(auc.test), Balanced_Accuracy=bacc, Sensitivity=sen, Specificity=spec, PPV=ppv, NPV=npv, F1=f1))
}

par(oma=c(0,0,0,0), xpd=F)
mycolors <- c("#111111", "#61D04F", "#2297E6")

## apply thresholds from ADNI2

# AUC for hippocampus residuals for whole sample
myroc.ad <- roc(Group~HP_vol_res, levels=c("CN","AD"), direction=">",
                data=subset(Hippocampus_values, subset=(Hippocampus_values$Group!="MCI") ))
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
                 data=subset(Hippocampus_values, subset=(Hippocampus_values$Group!="AD") ))# & (dat$Amy_status==dat$group_bin)))
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
