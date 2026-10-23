"""python -m unittest -v test_compare_patch; optional EXO_BASE_REFERENCE path."""
import ast
import math
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from compare_patch import patch_means, patch_intervals, paired_estimates
from lift_weekly import lead_estimator, instance_norm
from upstream.lead_estimate import estimate_indicator


def load_functions(path, names):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    env=dict(torch=torch,np=np,math=math)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_half_open_bins_and_clipped_tail(self):
        x=torch.arange(168.).reshape(1,1,168)
        torch.testing.assert_close(patch_means(x,12)[0,0],torch.arange(14.)*12+5.5)
        self.assertEqual(patch_intervals(168,12,12)[-1],(156,168))
        self.assertEqual(patch_intervals(168,11,7)[-1],(161,168))
        self.assertEqual(patch_means(x,11,7)[0,0,-1].item(),164.)

    def test_one_hour_identity(self):
        x=torch.randn(5,3,168,generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(patch_means(x,1),x))
        rl,rs,pl,ps=paired_estimates(x,1,1)
        self.assertTrue(torch.equal(rl,pl));self.assertTrue(torch.equal(rs,ps))

    def test_official_parity_at_raw_and_patch_lengths(self):
        for length in (3,14,15,168):
            x=torch.randn(13,3,length,generator=torch.Generator().manual_seed(length))
            lag,score,_=lead_estimator(x)
            ids,ref_lag,ref_score=estimate_indicator(instance_norm(x)[0],3)
            for j in (0,1):
                where=(ids[:,2]==j).long().argmax(-1)[:,None]
                self.assertTrue(torch.equal(lag[:,j],ref_lag[:,2].gather(-1,where)[:,0]))
                self.assertTrue(torch.equal(score[:,j],ref_score[:,2].gather(-1,where)[:,0]))

    def test_patch_delay_and_negative_sign(self):
        x=torch.randn(3,2,14,generator=torch.Generator().manual_seed(11))
        history=torch.cat([x,-x[:,:1].roll(2,-1)],1).repeat_interleave(12,-1)
        _,_,lag,score=paired_estimates(history,12,12)
        self.assertTrue((lag[:,0]==2).all())
        torch.testing.assert_close(score[:,0],-torch.ones(3))

    def test_within_patch_permutation_invariance(self):
        x=torch.randn(2,3,168,generator=torch.Generator().manual_seed(9))
        changed=x.reshape(2,3,14,12).flip(-1).reshape(2,3,168)
        torch.testing.assert_close(patch_means(x,12),patch_means(changed,12))
        a=lead_estimator(patch_means(x,12));b=lead_estimator(patch_means(changed,12))
        torch.testing.assert_close(a[0],b[0]);torch.testing.assert_close(a[1],b[1])

    def test_fallback_and_invalid_parameters(self):
        lag,score,_=lead_estimator(patch_means(torch.ones(2,3,168),12))
        self.assertTrue((lag==1).all());self.assertTrue((score==0).all())
        for width,stride in ((0,1),(12,0),(12,13)):
            with self.assertRaises(ValueError):patch_intervals(168,width,stride)
        with self.assertRaises(ValueError):lead_estimator(torch.ones(1,3,2))

    @unittest.skipUnless(os.environ.get('EXO_BASE_REFERENCE'),'Set EXO_BASE_REFERENCE for pinned-base parity')
    def test_base_packing_mean_and_fft_parity(self):
        root=Path(os.environ['EXO_BASE_REFERENCE'])
        pack=load_functions(root/'exobase/data.py',{'pack_patches'})['pack_patches']
        reduce=load_functions(root/'exobase/model.py',{'scalar_patch_means'})['scalar_patch_means']
        fft=load_functions(root/'exobase/lead_estimator.py',{'normalized_history','estimate_fft_leads'})['estimate_fft_leads']
        values=np.random.default_rng(17).normal(size=(168,3)).astype('float32')
        records=[('sample',np.arange(168)*3600,values,np.ones_like(values,dtype=bool))]
        history=torch.from_numpy(values.T.copy())[None]
        for width,stride in ((1,1),(12,12),(11,7)):
            data,_,mask=pack(records,0,168*3600,width*3600,list(range(3)),stride*3600)
            expected,valid=reduce(torch.from_numpy(data),torch.from_numpy(mask))
            actual=patch_means(history,width,stride)
            torch.testing.assert_close(actual,expected)
            base=fft(expected[:,2:,:,None],expected[:,:2,:,None],valid[:,2:],valid[:,:2],2)
            lag,score,_=lead_estimator(actual)
            torch.testing.assert_close(lag,base['lag'][:,0])
            torch.testing.assert_close(score,base['correlation'][:,0],atol=2e-7,rtol=1e-6)


if __name__=='__main__':unittest.main()
