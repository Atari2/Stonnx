use ndarray::{ArrayD, Ix2, SliceInfoElem};

use crate::common::{BoxResult, NDIndex};

pub fn matmul_impl<'b, A: Clone + num::Zero>(
    a: ndarray::ArrayViewD<'b, A>,
    b: ndarray::ArrayViewD<'b, A>,
) -> BoxResult<ArrayD<A>>
where
    for<'a> ndarray::ArrayView2<'a, A>:
        ndarray::linalg::Dot<ndarray::ArrayView2<'a, A>, Output = ndarray::Array2<A>>,
{
    let andim = a.ndim();
    let bndim = b.ndim();

    if andim == 2 && bndim == 2 {
        let a = a.into_dimensionality::<Ix2>()?;
        let b = b.into_dimensionality::<Ix2>()?;
        Ok(a.dot(&b).into_dyn())
    } else if andim == bndim && andim > 2 {
        let a_common_shape = &a.shape()[0..andim - 2];
        let b_common_shape = &b.shape()[0..bndim - 2];
        let last_2_a = &a.shape()[andim - 2..];
        let last_2_b = &b.shape()[bndim - 2..];
        if a_common_shape != b_common_shape {
            return Err(anyhow::anyhow!(
                "Matmul: a and b must have compatible shapes"
            ));
        }
        let out_shape = a_common_shape
            .iter()
            .chain(&last_2_a[0..1])
            .chain(&last_2_b[1..])
            .cloned()
            .collect::<Vec<_>>();
        let mut out = ArrayD::<A>::zeros(out_shape);
        for index in NDIndex::new(a_common_shape) {
            let asliceindex = index
                .iter()
                .copied()
                .map(|x| x.into())
                .chain([(..).into(), (..).into()])
                .collect::<Vec<SliceInfoElem>>();
            let aslice = a
                .slice(asliceindex.as_slice())
                .into_dimensionality::<Ix2>()?;
            let bslice = b
                .slice(asliceindex.as_slice())
                .into_dimensionality::<Ix2>()?;
            let mut outslice = out.slice_mut(asliceindex.as_slice());
            outslice.assign(&aslice.dot(&bslice));
        }
        Ok(out)
    } else {
        todo!(
            "Matmul not implemented for ndim {} and {}",
            a.ndim(),
            b.ndim()
        );
    }
}
