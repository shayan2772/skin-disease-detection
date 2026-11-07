/**
 * SkinCare AI — App Page Logic
 * Handles drag-and-drop upload, image analysis, and AI advice rendering.
 */

document.addEventListener('DOMContentLoaded', function () {

    // ── Guard: only run on the app page ──
    const dropZone = document.getElementById('drop-zone');
    if (!dropZone) return;

    // ── DOM References ──
    const fileInput      = document.getElementById('file-input');
    const browseBtn      = document.getElementById('browse-btn');
    const previewArea    = document.getElementById('preview-area');
    const previewImage   = document.getElementById('preview-image');
    const fileName       = document.getElementById('file-name');
    const fileSize       = document.getElementById('file-size');
    const removeBtn      = document.getElementById('remove-btn');
    const analyzeBtn     = document.getElementById('analyze-btn');
    const uploadSection  = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const predictionText = document.getElementById('prediction-text');
    const adviceLoading  = document.getElementById('advice-loading');
    const adviceContent  = document.getElementById('advice-content');
    const modelBadge     = document.getElementById('model-badge');
    const resetBtn       = document.getElementById('reset-btn');
    const downloadBtn    = document.getElementById('download-btn');
    const navNewAnalysis = document.getElementById('nav-new-analysis');
    const toastContainer = document.getElementById('toast-container');
    const resultImage        = document.getElementById('result-image');
    const confidenceValue    = document.getElementById('confidence-value');
    const confidenceBar      = document.getElementById('confidence-bar');
    const lowConfWarning     = document.getElementById('low-confidence-warning');
    const sourceBadge        = document.getElementById('source-badge');

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
    let selectedFile = null;

    // ────────────────────────────────────────
    // Drag & Drop
    // ────────────────────────────────────────
    ['dragenter', 'dragover'].forEach(function (evt) {
        dropZone.addEventListener(evt, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drop-highlight');
        });
    });

    dropZone.addEventListener('dragleave', function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drop-highlight');
    });

    dropZone.addEventListener('drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drop-highlight');
        var files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    });

    // ────────────────────────────────────────
    // Browse / Click to Upload
    // ────────────────────────────────────────
    dropZone.addEventListener('click', function (e) {
        // Avoid re-triggering if user clicked the browse button itself
        if (e.target === browseBtn || (browseBtn && browseBtn.contains(e.target))) return;
        if (fileInput) fileInput.click();
    });

    if (browseBtn) {
        browseBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            if (fileInput) fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function () {
            if (fileInput.files && fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });
    }

    // ────────────────────────────────────────
    // Handle File
    // ────────────────────────────────────────
    function handleFile(file) {
        if (!file) return;

        // Validate type
        if (!file.type || !file.type.startsWith('image/')) {
            showError('Please upload a valid image file (PNG, JPG, or WEBP).');
            return;
        }

        // Validate size
        if (file.size > MAX_FILE_SIZE) {
            showError('File is too large. Maximum size is 16 MB.');
            return;
        }

        selectedFile = file;

        // Read and show preview
        var reader = new FileReader();
        reader.onload = function (e) {
            if (previewImage) previewImage.src = e.target.result;
            if (fileName) fileName.textContent = file.name;
            if (fileSize) fileSize.textContent = formatFileSize(file.size);
            if (dropZone) dropZone.classList.add('hidden');
            if (previewArea) previewArea.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // ────────────────────────────────────────
    // Remove / Reset
    // ────────────────────────────────────────
    if (removeBtn) {
        removeBtn.addEventListener('click', function () {
            clearPreview();
        });
    }

    function clearPreview() {
        selectedFile = null;
        if (previewImage) previewImage.src = '';
        if (fileInput) fileInput.value = '';
        if (previewArea) previewArea.classList.add('hidden');
        if (dropZone) dropZone.classList.remove('hidden');
    }

    function resetAll() {
        clearPreview();
        if (resultsSection) resultsSection.classList.add('hidden');
        if (loadingSection) loadingSection.classList.add('hidden');
        if (uploadSection) uploadSection.classList.remove('hidden');
        if (predictionText) predictionText.textContent = '';
        if (adviceContent) {
            adviceContent.innerHTML = '';
            adviceContent.classList.add('hidden');
        }
        if (adviceLoading) adviceLoading.classList.remove('hidden');
        if (modelBadge) modelBadge.textContent = '';
        if (resultImage) resultImage.src = '';
        if (confidenceValue) confidenceValue.textContent = '';
        if (confidenceBar) { confidenceBar.style.width = '0%'; confidenceBar.className = 'h-2.5 rounded-full transition-all duration-700 ease-out'; }
        if (lowConfWarning) lowConfWarning.classList.add('hidden');
        if (sourceBadge) { sourceBadge.classList.add('hidden'); sourceBadge.textContent = ''; }
        if (navNewAnalysis) navNewAnalysis.classList.add('hidden');
        // Re-init AOS for fresh animations
        if (typeof AOS !== 'undefined') AOS.refresh();
    }

    if (resetBtn) resetBtn.addEventListener('click', resetAll);
    if (navNewAnalysis) navNewAnalysis.addEventListener('click', resetAll);

    // ────────────────────────────────────────
    // Analyze (Submit)
    // ────────────────────────────────────────
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function () {
            if (!selectedFile) {
                showError('Please select an image first.');
                return;
            }
            submitAnalysis();
        });
    }

    function submitAnalysis() {
        // Transition UI
        if (uploadSection) uploadSection.classList.add('hidden');
        if (loadingSection) loadingSection.classList.remove('hidden');

        var formData = new FormData();
        formData.append('file', selectedFile);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(function (response) {
            return response.json().then(function (data) {
                return { ok: response.ok, data: data };
            });
        })
        .then(function (result) {
            var data = result.data;

            if (data.error) {
                showError(data.error);
                backToUpload();
                return;
            }

            if (data.prediction) {
                showResults(data.prediction, data.confidence, data.source);
            } else {
                showError('Unexpected response from server.');
                backToUpload();
            }
        })
        .catch(function (err) {
            console.error('Prediction error:', err);
            showError('Something went wrong. Please try again.');
            backToUpload();
        });
    }

    function backToUpload() {
        if (loadingSection) loadingSection.classList.add('hidden');
        if (uploadSection) uploadSection.classList.remove('hidden');
    }

    function showResults(prediction, confidence, source) {
        if (loadingSection) loadingSection.classList.add('hidden');
        if (resultsSection) resultsSection.classList.remove('hidden');
        if (predictionText) predictionText.textContent = prediction;
        if (navNewAnalysis) navNewAnalysis.classList.remove('hidden');

        // Hide source badge (no branding)
        if (sourceBadge) sourceBadge.classList.add('hidden');

        // Show uploaded image in results
        if (resultImage && selectedFile) {
            var reader = new FileReader();
            reader.onload = function (e) {
                resultImage.src = e.target.result;
            };
            reader.readAsDataURL(selectedFile);
        }

        // Show confidence
        if (confidence !== undefined && confidence !== null) {
            if (confidenceValue) confidenceValue.textContent = confidence + '%';
            if (confidenceBar) {
                var barColor = confidence >= 70 ? 'bg-teal' : confidence >= 40 ? 'bg-amber-500' : 'bg-rose-500';
                confidenceBar.className = 'h-2.5 rounded-full transition-all duration-700 ease-out ' + barColor;
                setTimeout(function () { confidenceBar.style.width = confidence + '%'; }, 100);
            }
            if (lowConfWarning) {
                if (confidence < 50) {
                    lowConfWarning.classList.remove('hidden');
                } else {
                    lowConfWarning.classList.add('hidden');
                }
            }
        }

        // Reset advice panel to loading state
        if (adviceLoading) adviceLoading.classList.remove('hidden');
        if (adviceContent) {
            adviceContent.innerHTML = '';
            adviceContent.classList.add('hidden');
        }
        if (modelBadge) modelBadge.textContent = '';

        // Re-init AOS
        if (typeof AOS !== 'undefined') AOS.refresh();

        getAdvice(prediction, confidence);
    }

    // ────────────────────────────────────────
    // Get Advice
    // ────────────────────────────────────────
    function getAdvice(condition, confidence) {
        fetch('/advice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition: condition, confidence: confidence })
        })
        .then(function (response) {
            return response.json();
        })
        .then(function (data) {
            if (adviceLoading) adviceLoading.classList.add('hidden');
            if (adviceContent) adviceContent.classList.remove('hidden');

            if (data.error) {
                if (adviceContent) adviceContent.innerHTML = '<p class="text-warmgray italic">AI advice is currently unavailable. Please consult a dermatologist.</p>';
                return;
            }

            if (data.advice) {
                if (adviceContent) adviceContent.innerHTML = parseMarkdown(data.advice);
                if (modelBadge) modelBadge.textContent = '';
            } else {
                if (adviceContent) adviceContent.innerHTML = '<p class="text-warmgray italic">No advice returned. Please consult a dermatologist.</p>';
            }
        })
        .catch(function (err) {
            console.error('Advice error:', err);
            if (adviceLoading) adviceLoading.classList.add('hidden');
            if (adviceContent) {
                adviceContent.classList.remove('hidden');
                adviceContent.innerHTML = '<p class="text-warmgray italic">AI advice is currently unavailable. Please consult a dermatologist.</p>';
            }
        });
    }

    // ────────────────────────────────────────
    // Parse Markdown → HTML
    // ────────────────────────────────────────
    function parseMarkdown(text) {
        if (!text) return '';

        var html = '';
        var lines = text.split('\n');
        var inUl = false;
        var inOl = false;

        for (var i = 0; i < lines.length; i++) {
            var line = lines[i];

            // Heading ##
            if (/^##\s+(.+)/.test(line)) {
                if (inUl) { html += '</ul>'; inUl = false; }
                if (inOl) { html += '</ol>'; inOl = false; }
                html += '<h4>' + escapeHtml(line.replace(/^##\s+/, '')) + '</h4>';
                continue;
            }

            // Unordered list item
            if (/^[-*]\s+(.+)/.test(line)) {
                if (inOl) { html += '</ol>'; inOl = false; }
                if (!inUl) { html += '<ul>'; inUl = true; }
                var content = line.replace(/^[-*]\s+/, '');
                html += '<li>' + inlineFormat(content) + '</li>';
                continue;
            }

            // Ordered list item
            if (/^\d+\.\s+(.+)/.test(line)) {
                if (inUl) { html += '</ul>'; inUl = false; }
                if (!inOl) { html += '<ol>'; inOl = true; }
                var olContent = line.replace(/^\d+\.\s+/, '');
                html += '<li>' + inlineFormat(olContent) + '</li>';
                continue;
            }

            // Close any open lists
            if (inUl) { html += '</ul>'; inUl = false; }
            if (inOl) { html += '</ol>'; inOl = false; }

            // Empty line → paragraph break
            if (line.trim() === '') {
                html += '<br>';
                continue;
            }

            // Regular text
            html += '<p>' + inlineFormat(line) + '</p>';
        }

        // Close trailing lists
        if (inUl) html += '</ul>';
        if (inOl) html += '</ol>';

        return html;
    }

    function inlineFormat(text) {
        // Escape HTML first, then apply formatting
        var s = escapeHtml(text);
        // Bold: **text**
        s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        // Italic: *text* (single asterisk, not preceded by another *)
        s = s.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
        return s;
    }

    function escapeHtml(text) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    // ────────────────────────────────────────
    // Download Report (simple text download)
    // ────────────────────────────────────────
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function () {
            var prediction = predictionText ? predictionText.textContent : '';
            var adviceHtml = adviceContent ? adviceContent.innerHTML : '';
            if (!prediction) return;

            var conf = confidenceValue ? confidenceValue.textContent : '';
            var imgSrc = resultImage ? resultImage.src : '';
            var dateStr = new Date().toLocaleString();

            // Build styled HTML for PDF
            var html = '<div style="font-family: Georgia, serif; color: #1e293b; padding: 20px; max-width: 700px; margin: 0 auto;">';
            // Header
            html += '<div style="text-align: center; border-bottom: 3px solid #0d9488; padding-bottom: 20px; margin-bottom: 24px;">';
            html += '<h1 style="font-size: 24px; margin: 0 0 4px 0; color: #0d9488;">SkinCare AI</h1>';
            html += '<p style="font-size: 13px; color: #64748b; margin: 0;">Dermatology Consultation Report</p>';
            html += '</div>';
            // Patient info
            html += '<table style="width: 100%; font-size: 13px; margin-bottom: 20px; border-collapse: collapse;">';
            html += '<tr><td style="padding: 6px 0; color: #64748b; width: 120px;">Date</td><td style="padding: 6px 0; font-weight: 600;">' + dateStr + '</td></tr>';
            html += '<tr><td style="padding: 6px 0; color: #64748b;">Diagnosis</td><td style="padding: 6px 0; font-weight: 600; font-size: 16px; color: #0d9488;">' + prediction + '</td></tr>';
            if (conf) html += '<tr><td style="padding: 6px 0; color: #64748b;">Confidence</td><td style="padding: 6px 0; font-weight: 600;">' + conf + '</td></tr>';
            html += '</table>';
            // Image
            if (imgSrc) {
                html += '<div style="text-align: center; margin-bottom: 20px;">';
                html += '<img src="' + imgSrc + '" style="max-width: 200px; max-height: 200px; border-radius: 8px; border: 1px solid #e2e8f0;" />';
                html += '</div>';
            }
            // Advice
            html += '<div style="border-top: 2px solid #e2e8f0; padding-top: 16px; font-size: 13px; line-height: 1.7;">';
            html += '<h2 style="font-size: 16px; color: #0f172a; margin: 0 0 12px 0;">Dr. SkinCare AI — Clinical Report</h2>';
            html += adviceHtml;
            html += '</div>';
            // Disclaimer
            html += '<div style="margin-top: 24px; padding: 12px 16px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; font-size: 11px; color: #92400e;">';
            html += 'DISCLAIMER: This is an AI-generated report for informational purposes only. It is not a substitute for professional medical diagnosis or treatment. Please consult a board-certified dermatologist.';
            html += '</div>';
            html += '</div>';

            var container = document.createElement('div');
            container.innerHTML = html;
            document.body.appendChild(container);

            html2pdf().set({
                margin: [10, 10, 10, 10],
                filename: 'SkinCare-AI-Report-' + prediction.replace(/\s+/g, '-') + '.pdf',
                image: { type: 'jpeg', quality: 0.95 },
                html2canvas: { scale: 2, useCORS: true },
                jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
            }).from(container).save().then(function () {
                document.body.removeChild(container);
            });
        });
    }

    // ────────────────────────────────────────
    // Toast / Error Notification
    // ────────────────────────────────────────
    function showError(message) {
        if (!toastContainer) {
            alert(message);
            return;
        }

        var toast = document.createElement('div');
        toast.className = 'pointer-events-auto flex items-center gap-3 bg-rose-600 text-white px-5 py-3 rounded-xl shadow-lg text-sm font-medium toast-enter';
        toast.innerHTML =
            '<svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">' +
            '<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/>' +
            '</svg>' +
            '<span>' + escapeHtml(message) + '</span>';

        toastContainer.appendChild(toast);

        // Auto-dismiss after 5 seconds
        setTimeout(function () {
            toast.classList.remove('toast-enter');
            toast.classList.add('toast-exit');
            toast.addEventListener('animationend', function () {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            });
        }, 5000);
    }

    // ────────────────────────────────────────
    // Utility
    // ────────────────────────────────────────
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        var units = ['Bytes', 'KB', 'MB', 'GB'];
        var i = Math.floor(Math.log(bytes) / Math.log(1024));
        var size = (bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1);
        return size + ' ' + units[i];
    }

});
