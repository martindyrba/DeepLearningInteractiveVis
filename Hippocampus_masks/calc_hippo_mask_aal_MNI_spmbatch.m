%-----------------------------------------------------------------------
% Job saved on 28-Apr-2021 18:24:20 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7487)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.util.imcalc.input = {'C:\Users\DZNE\ADNI_Daten\aal\aal.nii,1'};
matlabbatch{1}.spm.util.imcalc.output = 'aal_hippocampus.nii';
matlabbatch{1}.spm.util.imcalc.outdir = {'C:\Users\DZNE\ADNI_Daten\Hippocampus_masks'};
matlabbatch{1}.spm.util.imcalc.expression = '(i1==37)+(i1==38)';
matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{1}.spm.util.imcalc.options.mask = 0;
matlabbatch{1}.spm.util.imcalc.options.interp = 0;
matlabbatch{1}.spm.util.imcalc.options.dtype = 2;
