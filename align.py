import subprocess


mapping = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
}

for iter in range(64):
    for idx in range(200):
        result = subprocess.run(
            f'calculate_rmsd ./generation/oapainn/iteration{iter}/{idx}/reaction_r.xyz \
            ./generation/oapainn/iteration{iter}/{idx}/reaction_p.xyz -e -p > \
            ./generation/oapainn/iteration{iter}/{idx}/reaction_p_align.xyz',
            capture_output=True,
            text=True,
            shell=True,
        )
        
        # convert number into elements
        with open(f'./generation/oapainn/iteration{iter}/{idx}/reaction_p_align.xyz', 'r') as fo:
            lines = fo.readlines()
        
        with open(f'./generation/oapainn/iteration{iter}/{idx}/reaction_p_align.xyz', 'w') as fo:
            fo.write(lines[0] + '\n')
            for line in lines[2:]:
                parts = line.split()
                if parts:  # ensure not empty
                    atomic_number = int(parts[0])  
                    if atomic_number in mapping:
                        parts[0] = mapping[atomic_number] 
                    fo.write(' '.join(parts) + '\n')
                    
    print(f'iteration {iter} done')
