citations =[['O governo canadense implantou aeronaves e helicópteros em apoio. O Ministro de Preparação para Emergências, Harjit Sajjan, declarou que o Canadá estava "pronto para apoiar nossos vizinhos americanos".', True], ['Bombeiros mexicanos chegaram ao Aeroporto Internacional de Los Angeles no dia 12 de janeiro de 2025 para ajudar no combate aos incêndios.', True], ['O Crescente Vermelho Iraniano ofereceu-se para enviar unidades de implantação rápida para combater incêndios florestais.', True], ['O presidente ucraniano, Volodymyr Zelensky, ofereceu-se para enviar bombeiros do Serviço de Emergência do Estado para a Califórnia para ajudar a lidar com os incêndios.', False], ['Em 15 de janeiro de 2025, o governo japonês forneceu US$ 2 milhões por meio da Cruz Vermelha para "estabelecer locais de evacuação e oferecer alimentos e apoio psicológico às pessoas afetadas pelos incêndios".', True]]
print(len(citations))
c = 0
for x in citations:
    if x[1] == True:
        c += 1

print(c)