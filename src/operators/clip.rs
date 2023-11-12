use super::super::utils::utils::ArrayType;


pub fn clip<'a>(
    inputs: &[ArrayType<'a>]
) -> Result<ArrayType<'a>, Box<dyn std::error::Error>> {
    let ilen = inputs.len();
    if ilen == 1 {
        return Ok(inputs[0].to_owned());
    }
    let amin = match inputs.get(1) {
        Some(a) => {
            if !a.shape().is_empty() {
                return Err("Amin must be a scalar".into());
            } else if let ArrayType::F32(a) = a {
                Some(a.sum())
            } else {
                todo!("CLIP amin not implemented for type {:?}", a)
            }
        },
        None => {
            None
        }
    };
    let amax = match inputs.get(2) {
        Some(a) => {
            if !a.shape().is_empty() {
                return Err("Amax must be a scalar".into());
            } else if let ArrayType::F32(a) = a {
                Some(a.sum())
            } else {
                todo!("CLIP amax not implemented for type {:?}", a)
            }
        },
        None => {
            None
        }
    };
    if let ArrayType::F32(a) = &inputs[0] {
        let mut a = a.to_owned();
        if let Some(amin) = amin {
            a.mapv_inplace(|v| v.max(amin));
        }
        if let Some(amax) = amax {
            a.mapv_inplace(|v| v.min(amax));
        }
        return Ok(ArrayType::OwnF32(a));
    } else {
        todo!("CLIP not implemented for type {:?}", inputs[0])
    }
}