from einops import rearrange

# define reshape methods for different input

methods = {
        'image':
            {
                'reshape':
                [
                    'h w c -> 1 c h w ',
                    '1 c h w -> h w c',
                    '1 c (n2 p2) (n1 p1) -> (n2 n1) (p2 p1) c',
                    '(n2 p2) (n1 p1) c -> (n2 n1) (p2 p1) c',
                    '(n2 n1) (p2 n1) c -> 1 c (n2 p2) (n1 p1)',
                    '(n2 n1) -> 1 n2 n1',
                    '1 n2 n1 -> 1 (n2 p2) (n1 p1)',
                    '1 h w c -> 1 (h w) c'
                ],
                'mode': 'bilinear'
            },
        'mesh':
            {
                'reshape':
                [
                    'd h w c -> 1 c d h w',
                    '1 c d h w -> d h w c',
                    '1 c (n3 p3) (n2 p2) (n1 p1) -> (n3 n2 n1) (p3 p2 p1) c',
                    '(n3 p3) (n2 p2) (n1 p1) c -> (n3 n2 n1) (p3 p2 p1) c',
                    '(n3 n2 n1) (p3 p2 p1) c -> 1 c (n3 p3) (n2 p2) (n1 p1)',
                    '(n3 n2 n1) -> 1 n3 n2 n1',
                    '1 n3 n2 n1 -> 1 (n3 p3) (n2 p2) (n1 p1)',
                    '1 d h w c -> 1 (d h w) c',
                    ],
                    'mode': 'trilinear'
                }
    }

def einops_f(x, method, hparams=None, f=rearrange):
    """
    Apply einops operation @f on @x according to @method
    with params in namespace @hparams
    Filter out unused params before passing to @f
    """
    if hparms is None:
        return f(x, method)

    required_keys = set(method.replace('(', '').replace(')', '').split(' '))

    return f(x, method, **{k: v 
                            for k, v in vars(hparams).items() 
                            if k in required_keys})
