document.addEventListener('DOMContentLoaded', () => {
	const form = document.getElementById('queryForm')
	const loading = document.getElementById('loading')
	const responseContainer = document.getElementById('response')
	const responseText = responseContainer.querySelector('#response-text')
	const sources = responseContainer.querySelector('#sources')

	form.addEventListener('submit', async (event) => {
		event.preventDefault();
		const formData = new FormData(form);
		loading.classList.remove('hidden')
		try {
			const response = await fetch('/query', {
				method: 'POST',
				body: formData
			});
			const data = await response.json();
			console.log(data);
			responseContainer.classList.remove('hidden')
			responseText.textContent = data.response
			sources.innerHTML = data.sources.map(source => 
				`<div class="source"><h4>${source.id.replace('app/data/', '')}:</h4><p>${source.page_content}</p></div>`
			).join('')
		} catch (error) {
			console.error('Error:', error)
			responseContainer.classList.add('hidden')
		}
		loading.classList.add('hidden')
	});
});