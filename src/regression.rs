/// calculate the Theil-Sen and Mann-Kendall statistics
pub fn mk(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    assert!(xs.len() == ys.len());
    let n = xs.len();
    let triulen = n * (n - 1) / 2;
    let mut slopes = Vec::<f64>::with_capacity(triulen);
    let mut s = 0f64;
    for i in 0..n {
        for j in 0..i {
            let x = xs[i] - xs[j];
            let y = ys[i] - ys[j];
            slopes.push(y / x);
            let v = x * y;
            if v != 0. { s += v.signum(); }
        }
    }

    let l = slopes.len();
    let median_slope = if l % 2 == 0 {
        (*order_stat::kth_by(&mut slopes, l / 2 - 1, |u, v| u.partial_cmp(v).unwrap()) +
        *order_stat::kth_by(&mut slopes, l / 2, |u, v| u.partial_cmp(v).unwrap())) / 2.
    } else {
        *order_stat::kth_by(&mut slopes, l / 2, |u, v| u.partial_cmp(v).unwrap())
    };

    // group ys by value
    let mut ycs = std::collections::HashMap::<BitHashedF64, usize>::new();
    for y in ys { *ycs.entry(BitHashedF64{0: *y}).or_insert(0) += 1; }

    // extract sizes of groups of same-valued ys larger than 1
    let ties = ycs.into_iter()
        .filter_map(|(_v, c)| {
            if c > 1 { Some(c) }
            else { None }
        }).collect::<Vec<_>>();
    // group the groups by their sizes
    let mut group_sizes = std::collections::HashMap::<usize, usize>::new();
    for group_size in ties {
        *group_sizes.entry(group_size).or_insert(0) += 1;
    }

    let adj: usize = group_sizes.iter().map(|(group_size, count)| {
        count * group_size * (group_size-1) * (2*group_size+5)
    }).sum();

    let v = n * (n-1) * (2*n+5) - adj;
    let v = v as f64/18.;

    let z = if s > 0. {
        (s-1.) / v.sqrt()
    } else if s < 0. {
        (s+1.) / v.sqrt()
    } else {
        0.
    };

    let standard_normal = statrs::distribution::Normal::new(0., 1.).unwrap();
    use statrs::distribution::ContinuousCDF;
    let p = 2.*(1.-standard_normal.cdf(z.abs()));

    (p, median_slope)
}

/// calculate the Theil-Sen and Mann-Kendall statistics and intercept
pub fn mk_intercept(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let (p, median_slope) = mk(xs, ys);
    let mut vs = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| y - median_slope * x)
        .collect::<Vec<f64>>();
    let l = vs.len();
    let median_intercept = if l % 2 == 0 {
        (*order_stat::kth_by(&mut vs, l / 2 - 1, |u, v| u.partial_cmp(v).unwrap()) +
        *order_stat::kth_by(&mut vs, l / 2, |u, v| u.partial_cmp(v).unwrap())) / 2.
    } else {
        *order_stat::kth_by(&mut vs, l / 2, |u, v| u.partial_cmp(v).unwrap())
    };

    (p, median_slope, median_intercept)
}

fn mean(vs: &[f64]) -> f64 { vs.iter().sum::<f64>() / vs.len() as f64 }

/// Calculate the Simple Linear Regression p-value and slope and intercept
pub fn linreg_intercept(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    assert!(xs.len() == ys.len());
    let n = xs.len();
    assert!(n >= 2);

    let meany = mean(&ys);
    let meanx = mean(&xs);

    let btop = std::iter::zip(xs.iter(), ys.iter())
        .map(|(x, y)| (y - meany)*(x - meanx))
        .sum::<f64>();
    let bbot = xs.iter()
        .map(|x| (x - meanx) * (x - meanx))
        .sum::<f64>();
    let b = btop / bbot;

    let a = meany - b*meanx;

    let evaluated = xs.iter()
        .map(|x| b*x + a)
        .collect::<Vec<f64>>();
    let residuals = std::iter::zip(evaluated.iter(), ys.iter())
        .map(|(e, y)| e - y)
        .collect::<Vec<f64>>();

    let p = if n == 2 {
        0.
    } else {
        let var = residuals.iter()
            .map(|r| r*r)
            .sum::<f64>()
            /
            (n - 2) as f64;
        let rad = var /
            xs.iter()
            .map(|x| (x - meanx)*(x - meanx))
            .sum::<f64>();

        if rad <= 0. {
            0.
        } else {
            let rads = rad.sqrt();
            let t = b / rads;
            let tdist = statrs::distribution::StudentsT::new(0., 1., (n - 2) as f64).unwrap();
            use statrs::distribution::ContinuousCDF;
            2. * (1. - tdist.cdf(t.abs()))
        }
    };
    (p, b, a)
}

/// Calculate the Simple Linear Regression p-value and slope
pub fn linreg(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let (p, b, _a) = linreg_intercept(xs, ys);
    (p, b)
}

struct BitHashedF64 (f64);
impl std::hash::Hash for BitHashedF64 {
    fn hash<H>(&self, state: &mut H)
        where H: std::hash::Hasher
    { self.0.to_bits().hash(state); }
}
impl std::cmp::PartialEq for BitHashedF64 {
    fn eq(&self, other: &Self) -> bool { self.0.to_bits() == other.0.to_bits() }
}
impl std::cmp::Eq for BitHashedF64 {}


#[cfg(test)]
mod tests {
    use crate::regression::*;
    /*#[test]
    fn test_cmp_numeric() {
        assert!(cmp_numeric(&vec![1], &vec![2]).is_lt());
        assert!(cmp_numeric(&vec![2], &vec![11]).is_lt());
        assert!(cmp_numeric(&vec![0,0,2], &vec![11]).is_lt());
    }*/

    #[test]
    fn test_linreg() {
        assert_eq!(linreg(&[0.,1.,2.,3.,4.], &[0., 2., 4., 6., 8.]), (0.0, 2.0));
        assert_eq!(linreg(&[0.,1.,2.,3.,4.,5.], &[0., 2., 4., 6., 8.,10.]), (0.0, 2.0));
        assert_eq!(linreg(&[0.,1.,2.,3.,4.], &[1., 2., 3., 4., 5.]), (0.0, 1.0));
        assert_eq!(linreg(&[0.,1.,2.,3.,4.], &[1., 1., 1., 1., 1.]), (0.0, 0.0));
        assert_eq!(linreg(&[0.,1.,2.,3.,4.], &[1., 1., 1., 1., 2.]), (0.1816901138162088, 0.2));
    }

    #[test]
    fn test_linreg_flat() {
        assert_eq!(linreg(&[0.,1.,2.,3.,4.], &[0.,0.,0.0004,0.,0.]), (1.0, 2.7105054312137612e-21));
    }

    #[test]
    fn test_linreg_intercept() {
        assert_eq!(linreg_intercept(&[0.,1.,2.,3.,4.], &[1.,3.,5.,7.,9.]), (0.0, 2.0, 1.0));
        assert_eq!(linreg_intercept(&[0.,1.,2.,3.,4.], &[1.,1.,1.,1.,1.]), (0.0, 0.0, 1.0));
        assert_eq!(linreg_intercept(&[0.,1.,2.,3.,4.], &[2.,4.,6.,8.,10.]), (0.0, 2.0, 2.0));
        assert_eq!(linreg_intercept(&[0.,1.,2.,3.,4.], &[0.,2.,4.,6.,8.]), (0.0, 2.0, 0.0));
        assert_eq!(linreg_intercept(&[0.,1.,2.,3.,4.], &[1.,0.,-1.,-2.,-3.]), (0.0, -1.0, 1.0));
    }

    #[test]
    fn test_mk() {
        assert_eq!(mk(&[0.,1.,2.,3.,4.], &[1., 1., 1., 1., 1.]), (1.0, 0.0));
        assert_eq!(mk(&[0.,1.,2.,3.,4.], &[1., 2., 3., 4., 5.]), (0.027486336110310372, 1.0));
        assert_eq!(mk(&[0.,1.,2.,3.,4.], &[0., 2., 4., 6., 8.]), (0.027486336110310372, 2.0));
        assert_eq!(mk(&[0.,1.,2.,3.,4.], &[1., 1., 1., 1., 2.]), (0.288844366370576, 0.0));
        assert_eq!(mk(&[0.,1.,2.,3.,4.,5.], &[0., 2., 4., 6., 8.,10.]), (0.008534920413867608, 2.0));
    }

    #[test]
    fn test_mk2() {
        assert_eq!(mk(&[0.,1.,2.,3.,4.,5.,6.,7.,8.], &[23., 24., 29., 6., 29., 24., 24., 29., 23.]), (0.8269210217567053, 0.0));
    }

    #[test]
    fn test_mk_intercept() {
        assert_eq!(mk_intercept(&[0.,1.,2.,3.,4.], &[1.,3.,5.,7.,9.]), (0.027486336110310372, 2.0, 1.0));
        assert_eq!(mk_intercept(&[0.,1.,2.,3.,4.], &[1.,1.,1.,1.,1.]), (1.0, 0.0, 1.0));
        assert_eq!(mk_intercept(&[0.,1.,2.,3.,4.], &[2.,4.,6.,8.,10.]), (0.027486336110310372, 2.0, 2.0));
        assert_eq!(mk_intercept(&[0.,1.,2.,3.,4.], &[0.,2.,4.,6.,8.]), (0.027486336110310372, 2.0, 0.0));
        assert_eq!(mk_intercept(&[0.,1.,2.,3.,4.], &[1.,0.,-1.,-2.,-3.]), (0.027486336110310372, -1.0, 1.0));
    }

    /*
    #[test]
    fn test_tag_ordering() {
        let tags = vec!["2020-008", "2022-01", "2019", "1982-04-02", "2018-02"];
        let ord = tag_ordering(&tags);
        assert_eq!(ord,
            [(3, "1982-04-02", vec![1982, 4, 2]),
             (4, "2018-02",    vec![2018, 2]),
             (2, "2019",       vec![2019]),
             (0, "2020-008",   vec![2020, 8]),
             (1, "2022-01",    vec![2022, 1])]);
    }*/
}
