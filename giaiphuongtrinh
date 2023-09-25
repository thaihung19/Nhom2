import numpy as np

def giai_he_phuong_trinh(A, b):
    n = len(A)
    x = np.zeros(n)

    # Chuyển ma trận A và vector b thành một ma trận mở rộng
    augmented_matrix = np.column_stack((A, b))

    # Áp dụng phương pháp khử Gauss để chuyển ma trận mở rộng về dạng tam giác trên
    for i in range(n):
        max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
        for j in range(i + 1, n):
            ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, :] -= ratio * augmented_matrix[i, :]

    # Giải hệ phương trình bằng phương pháp lùi
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], x[i+1:n])) / augmented_matrix[i, i]

    return x

# Ví dụ: giải hệ phương trình 3x + 2y - z = 1 và 2x - y + 3z = -2
A = np.array([[3, 2, -1], [2, -1, 3]])
b = np.array([1, -2])

solution = giai_he_phuong_trinh(A, b)
print("Nghiệm của hệ phương trình là:", solution)
for i, x in enumerate(solution):
    print(f"x_{i + 1} = {x}")
