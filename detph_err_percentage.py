import os
from tqdm import tqdm
import matplotlib.pyplot as plt  # 新增匯入繪圖模組

def read_txt(file_path):
    """讀取 TXT 檔案並解析內容"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        cls, bbox = parts[0], list(map(float, parts[1:]))
        data.append((cls, bbox))
    return data

def calculate_iou(box1, box2):
    """計算兩個 Bounding Box 的 IoU"""
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2

    # 計算交集
    xi1, yi1 = max(x1 - w1 / 2, x2 - w2 / 2), max(y1 - h1 / 2, y2 - h2 / 2)
    xi2, yi2 = min(x1 + w1 / 2, x2 + w2 / 2), min(y1 + h1 / 2, y2 + h2 / 2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 計算聯集
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def calculate_depth_error(gt_depth, det_depth):
    """計算深度值的誤差百分比"""
    if gt_depth == 0:  # 避免除以零
        return float('inf')
    return abs(gt_depth - det_depth) / gt_depth * 100

def compare_bounding_boxes(gt_file, det_file, bounduary):
    """比對 Ground Truth 和 Detect 的 Bounding Box，並計算深度誤差"""
    gt_data = read_txt(gt_file)
    det_data = read_txt(det_file)

    results = []
    tmp = {}
    is_save = False
    for gt_cls, gt_box in gt_data:
        best_iou = 0.5
        best_det = None
        for det_cls, det_box in det_data:
            if gt_cls == det_cls:  # 類別需一致
                iou = calculate_iou(gt_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det_box
        if best_det:
            gt_depth = gt_box[-1]  # Ground Truth 的深度值
            det_depth = best_det[-1]  # Detect 的深度值
            depth_error = calculate_depth_error(gt_depth, det_depth)
            # if depth_error >= bounduary:
            #     print(gt_cls)
            #     print(gt_box)
            #     print(best_det)
            #     print('-----')

            if bounduary == 20 and depth_error >= bounduary and not is_save:
                with open('check.txt', 'a') as f:
                    f.write(f"{gt_file.split('/')[-1].replace('.txt', '.png')}\n")
        else:
            depth_error = None
            continue
        results.append((gt_cls, gt_box, best_det, best_iou, depth_error))
    return results

def plot_all_depth_errors_scatter(all_results, output_path, bounduary):
    """繪製所有檔案的深度誤差百分比點圖並儲存為檔案，並計算 < 20% 的數量"""
    x_low = [] 
    y_low = []  
    x_high = []  
    y_high = []  

    for results in all_results:
        for _, gt_box, _, _, depth_error in results:
            if depth_error is not None:
                if depth_error < bounduary:
                    x_low.append(gt_box[-1])
                    y_low.append(depth_error)
                else:
                    x_high.append(gt_box[-1])
                    y_high.append(depth_error)

    # 計算 < bounduary% 的數量
    low_count = len(x_low)
    print(f"Number of points with Depth Error < {bounduary}%: {low_count}")
    high_count = len(x_high)
    print(f"Number of points with Depth Error >= {bounduary} %: {high_count}")
    print(f"Depth Error < {bounduary}% ratio:{low_count / (low_count + high_count) * 100}%")

    plt.figure(figsize=(10, 6))
    plt.scatter(x_low, y_low, s=10, alpha=0.6, color='green', label=f'Depth Error < {bounduary}%')  # 綠色點
    plt.scatter(x_high, y_high, s=10, alpha=0.6, color='red', label=f'Depth Error >= {bounduary}%')  # 紅色點
    plt.title("Depth Error Percentage vs Ground Truth Depth (All Files)")
    plt.xlabel("Ground Truth Depth")
    plt.ylabel("Depth Error Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)  # 儲存圖表
    plt.close()

def plot_depth_difference_scatter(all_results, output_path):
    """繪製 x 軸為 Ground Truth Depth，y 軸為 Ground Truth - Detect 差值的點圖"""
    x = []  # 真實深度值
    y = []  # 差值 (Ground Truth - Detect)

    for results in all_results:
        for _, gt_box, det_box, _, _ in results:
            if det_box is not None:
                x.append(gt_box[-1])  # Ground Truth Depth
                y.append(gt_box[-1] - det_box[-1])  # 差值 (Ground Truth - Detect)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=10, alpha=0.6, color='blue', label='Depth Difference (GT - Detect)')
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Zero Difference')  # 加一條 y=0 的水平線
    plt.title("Depth Difference vs Ground Truth Depth")
    plt.xlabel("Ground Truth Depth")
    plt.ylabel("Depth Difference (GT - Detect)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)  # 儲存圖表
    plt.close()

def plot_ratio_bar_chart(bounduary_results, output_path):
    """繪製不同 bounduary 值下 low_count 和 high_count 的占比柱狀圖，並在柱狀圖上顯示百分比"""
    bounduaries = []
    low_ratios = []
    high_ratios = []

    for bounduary, (low_count, high_count) in bounduary_results.items():
        bounduaries.append(bounduary)
        total = low_count + high_count
        low_ratios.append(low_count / total * 100)
        high_ratios.append(high_count / total * 100)

    x = range(len(bounduaries))
    plt.figure(figsize=(10, 6))
    plt.bar(x, low_ratios, color='green', label='Depth Error < Bounduary (%)')
    plt.bar(x, high_ratios, bottom=low_ratios, color='red', label='Depth Error >= Bounduary (%)')

    # 在柱狀圖上顯示百分比
    for i, (low, high) in enumerate(zip(low_ratios, high_ratios)):
        plt.text(i, low / 2, f"{low:.1f}%", ha='center', va='center', color='white', fontsize=10)  # 綠色部分
        plt.text(i, low + high / 2, f"{high:.1f}%", ha='center', va='center', color='white', fontsize=10)  # 紅色部分

    plt.xticks(x, [f"{b}%" for b in bounduaries])
    plt.xlabel("Bounduary (%)")
    plt.ylabel("Percentage (%)")
    plt.title("Low and High Depth Error Ratios for Different Bounduaries")
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(output_path)
    plt.close()

def main():
    # detect_dir = os.path.join(os.getcwd(), 'detect', '5ch11/labels')
    detect_dir = "/mnt/hdd8t/kj-chang/yolov5detect/5ch+depth/5ch12/labels"
    output_dir = os.path.join(os.getcwd(), 'output')  # 儲存圖表的目錄
    os.makedirs(output_dir, exist_ok=True)

    all_results = []  # 儲存所有檔案的結果
    bounduary_results = {}  # 儲存不同 bounduary 的 low_count 和 high_count

    file_paths = sorted(os.scandir(detect_dir), key=lambda x: x.name)
    for bounduary in [1] + list(range(2, 22, 2)):  # 測試不同的 bounduary 值，包含 1
        low_count = 0
        high_count = 0
        for i, fp in enumerate(tqdm(file_paths)):
            gt_file = os.path.join('/mnt/ssd/kj-chang/traindata/sourcedata/5-channel/labels', fp.name)
            det_file = os.path.join(detect_dir, fp.name)

            results = compare_bounding_boxes(gt_file, det_file, bounduary)
            all_results.append(results)

            for _, _, _, _, depth_error in results:
                if depth_error is not None:
                    if depth_error < bounduary:
                        low_count += 1
                    else:
                        high_count += 1

        bounduary_results[bounduary] = (low_count, high_count)

        # 繪製所有檔案的點圖
        output_path_scatter = os.path.join(output_dir, f"all_files_depth_error_bounduary{bounduary}.png")
        plot_all_depth_errors_scatter(all_results, output_path_scatter, bounduary)

    # 繪製 Ground Truth - Detect 差值的圖表
    output_path_difference = os.path.join(output_dir, "depth_difference_scatter.png")
    plot_depth_difference_scatter(all_results, output_path_difference)

    # 繪製柱狀圖
    output_path_bar = os.path.join(output_dir, "depth_error_ratios_bar_chart.png")
    plot_ratio_bar_chart(bounduary_results, output_path_bar)

if __name__ == "__main__":
    main()
